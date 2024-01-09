import torch, random, numpy as np, torch.nn.functional as F
from bidict import bidict
from collections import defaultdict
from torch import nn
from cky_parser_sgd import BatchCKYParser
from char_coding_models import CharProbRNN, ResidualLayer, \
    WordProbFCFixVocabCompound
from cg_type import CGNode, generate_categories_by_depth, \
    read_categories_from_file, get_category_argument_depths

QUASI_INF = 10000000.

DEBUG = False
def printDebug(*args, **kwargs):
    if DEBUG:
        print("DEBUG: ", end="")
        print(*args, **kwargs)


class Predicate:
    def __init__(self, word, pos):
        self.word = word
        self.pos = pos

    def __str__(self):
        return "{}_{}".format(self.word, self.pos)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return str(self) < str(other)


class BasicCGInducer(nn.Module):
    def __init__(self, config, num_chars, num_words):
        super(BasicCGInducer, self).__init__()
        self.state_dim = config.getint("state_dim")
        self.rnn_hidden_dim = config.getint("rnn_hidden_dim")
        self.model_type = config["model_type"]


        self.device = config["device"]
        self.eval_device = config["eval_device"]

        if self.model_type == 'char':
            self.emit_prob_model = CharProbRNN(
                num_chars,
                state_dim=self.state_dim,
                hidden_size=self.rnn_hidden_dim
            )
        elif self.model_type == 'word':
            self.emit_prob_model = WordProbFCFixVocabCompound(
                num_words, self.state_dim
            )
        else:
            raise ValueError("Model type should be char or word")

        self.num_primitives = config.getint("num_primitives", fallback=None)
        self.max_func_depth = config.getint("max_func_depth", fallback=None)
        self.max_arg_depth = config.getint("max_arg_depth", fallback=None)
        self.cats_list = config.get("category_list", fallback=None)
        self.arg_depth_penalty = config.getfloat(
            "arg_depth_penalty", fallback=None
        )
        self.left_arg_penalty = config.getfloat(
            "left_arg_penalty", fallback=None
        )

        # option 1: specify a set of categories according to maximum depth
        # and number of primitives (used in 2023 ACL Findings paper)
        if self.num_primitives is not None:
            assert self.max_func_depth is not None
            assert self.cats_list is None
            if self.max_arg_depth is None:
                self.max_arg_depth = self.max_func_depth - 1
            self.init_cats_by_depth()

        # option 2: specify a file with a list of categories
        else:
            assert self.cats_list is not None
            # all of this stuff is just for option 1
            assert self.max_func_depth is None
            assert self.max_arg_depth is None
            assert self.arg_depth_penalty is None
            assert self.left_arg_penalty is None
            self.init_cats_from_list()

        # cat_arg_depths[i] gives the number of arguments taken by category
        # ix2cat[i]
        self.cat_arg_depths = get_category_argument_depths(self.ix2cat)

        # arg_depth_to_cats[d] gives the set of categories (indices) with arg depth d
        self.arg_depth_to_cats_res = defaultdict(set)
        self.arg_depth_to_cats_arg = defaultdict(set)
        for ix, depth in self.cat_arg_depths.items():
            if self.ix2cat[ix] in self.res_cats:
                self.arg_depth_to_cats_res[depth].add(ix)
            if self.ix2cat[ix] in self.arg_cats:
                self.arg_depth_to_cats_arg[depth].add(ix)


        self.init_predicates(config)
        self.init_predcats()
        self.init_association_matrix()
        self.init_masks()
        self.init_functor_lookup_tables()

        printDebug("cat_arg_depths:", self.cat_arg_depths)
        printDebug("arg_depth_to_cats_res:", self.arg_depth_to_cats_res)
        printDebug("arg_depth_to_cats_arg:", self.arg_depth_to_cats_arg)
        printDebug("ix2cat:", self.ix2cat)
        printDebug("res_cats:", self.res_cats)
        printDebug("arg_cats:", self.arg_cats)
        printDebug("ix2pred:", self.ix2pred)

        # CG: "embeddings" for the categories are just one-hot vectors
        # these are used for result categories
        self.fake_emb = nn.Parameter(
            torch.eye(self.qres)
        )
        state_dim = self.state_dim
        # actual embeddings are used to calculate split scores
        # (i.e. prob of terminal vs nonterminal)
        self.nt_emb = nn.Parameter(
            torch.randn(self.qall, state_dim)
        )
        # embeddings for predicate-category pairs
        self.predcat_emb = nn.Parameter(
            torch.randn(self.qall, state_dim)
        )
        # maps res_cat to arg_cat x {arg_on_L, arg_on_R}
        self.rule_mlp = nn.Linear(
            self.qres,
            2*self.qarg
        )

        self.root_emb = nn.Parameter(torch.eye(1)).to(self.device)
        self.root_mlp = nn.Linear(1, self.qall).to(self.device)

        # decides terminal or nonterminal
        self.split_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            nn.Linear(state_dim, 2)
        ).to(self.device)

        self.parser = BatchCKYParser(
            ix2cat=self.ix2cat,
            ix2pred=self.ix2pred,
            ix2predcat=self.ix2predcat,
            ix2predcat_res=self.ix2predcat_res,
            ix2predcat_arg=self.ix2predcat_arg,
            argpc_2_pc = self.argpc_2_pc,
            lfunc_ixs=self.lfunc_ixs,
            rfunc_ixs=self.rfunc_ixs,
            larg_mask=self.larg_mask,
            rarg_mask=self.rarg_mask,
            qall=self.qall,
            qres=self.qres,
            qarg=self.qarg,
            device=self.device
        )


    def init_cats_by_depth(self):
        cats_by_max_depth, ix2cat, ix2depth = generate_categories_by_depth(
            self.num_primitives,
            self.max_func_depth,
            self.max_arg_depth
        )
        all_cats = cats_by_max_depth[self.max_func_depth]
        # res_cats (result cats) are the categories that can be
        # a result from a functor applying to its argument.
        # Since a functor's depth is one greater than the max depth between
        # its argument and result, the max depth of a result
        # is self.max_func_depth-1
        res_cats = cats_by_max_depth[self.max_func_depth-1]
        # optionally constrain the complexity of argument categories
        arg_cats = cats_by_max_depth[self.max_arg_depth]

        self.all_cats = sorted(all_cats)
        self.res_cats = sorted(res_cats)
        self.arg_cats = sorted(arg_cats)
        self.ix2cat = ix2cat

        # WARNING: this assumes that the argument categories are the first
        # len(arg_cats) categories in the full set (which may not be true)
        if self.arg_depth_penalty:
            printDebug("ix2depth: {}".format(ix2depth))
            ix2depth_arg = torch.Tensor(ix2depth[:len(arg_cats)])
            # dim: Qres x 2Qarg
            arg_penalty_mat = -self.arg_depth_penalty \
                * ix2depth_arg.tile((len(res_cats), 2))
            self.arg_penalty_mat = arg_penalty_mat.to(self.device)


    def init_cats_from_list(self):
        all_cats, res_cats, arg_cats, ix2cat = read_categories_from_file(
            self.cats_list
        )
        self.all_cats = sorted(all_cats)
        self.res_cats = sorted(res_cats)
        self.arg_cats = sorted(arg_cats)
        self.ix2cat = ix2cat


    def init_predicates(self, config):
        predicates = set()
        # pred_role_counts[p] is the number of roles associated with
        # predicate p
        pred_role_counts = defaultdict(int)
        # arg1_assoc[h][j] is the weight for j being h's first argument
        arg1_assoc = defaultdict(dict)
        arg2_assoc = defaultdict(dict)

        f_assoc = open(config["predicate_associations"])
        # header
        f_assoc.readline()
        for l in f_assoc:
            pred1, pred1role, pred2, pred2role, score = l.strip().split()
            pred1role = int(pred1role)
            pred2role = int(pred2role)
            score = float(score)
            predicates.add(pred1)
            predicates.add(pred2)
            pred_role_counts[pred1] = max(pred1role, pred_role_counts[pred1])
            pred_role_counts[pred2] = max(pred2role, pred_role_counts[pred2])
            if pred1role == 0 and pred2role == 1:
                # NOTE: reverse the order of the predicates so that the thing
                # taking the argument comes first
                arg1_assoc[pred2][pred1] = score
                #entries_arg1.append((pred2, pred1, score))
            elif pred1role == 0 and pred2role == 2:
                arg2_assoc[pred2][pred1] = score
            else:
                raise NotImplementedError(
                    "Only first and second arguments are currently supported. Offending line: {}".format(l)
                )

        preds_by_role_count = defaultdict(set)
        for p in pred_role_counts:
            count = pred_role_counts[p]
            preds_by_role_count[count].add(p)

        ix2pred = bidict(enumerate(sorted(predicates)))

        self.ix2pred = ix2pred
        self.preds_by_role_count = preds_by_role_count
        self.pred_role_counts = pred_role_counts
        self.arg1_assoc = arg1_assoc
        self.arg2_assoc = arg2_assoc


    def init_predcats(self):
        max_pred_role_count = max(self.preds_by_role_count)
        # ix2predcat assigns an index to each valid (predicate, category)
        # pair. E.g., if the index of (DOG, 1/0) is 5, then
        # ix2predcat[5] = (ix2pred.inv[DOG], ix2pred.inv[1/0])
        ix2predcat = bidict()
        # preds_per_cat[i] is the number of different predicates that
        # could be assigned to category ix2cat[i]
        preds_per_cat = defaultdict(set)
        for c in self.all_cats:
            c_ix = self.ix2cat.inv[c]
            arg_depth = self.cat_arg_depths[c_ix]
            for i in range(arg_depth, max_pred_role_count+1):
                preds_with_i_args = self.preds_by_role_count[i]
                for p in preds_with_i_args:
                    p_ix = self.ix2pred.inv[p]
                    ix2predcat[len(ix2predcat)] = (p_ix, c_ix)
                    preds_per_cat[c_ix].add(p)

        ix2predcat_res = bidict()
        res_preds_per_cat = defaultdict(set)
        for c in self.res_cats:
            c_ix = self.ix2cat.inverse[c]
            arg_depth = self.cat_arg_depths[c_ix]
            # NOTE: a predicate must have at least n+1 roles to appear with a
            # category of arg depth n. This means that e.g. DOG cannot appear as a
            # result category even with a primitive category. This rule may need to
            # be changed when modification is added
            for i in range(arg_depth+1, max_pred_role_count+1):
                preds_with_i_args = self.preds_by_role_count[i]
                for p in preds_with_i_args:
                    p_ix = self.ix2pred.inv[p]
                    ix2predcat_res[len(ix2predcat_res)] = (p_ix, c_ix)
                    res_preds_per_cat[c_ix].add(p)
        printDebug("ix2predcat_res:", ix2predcat_res)


        ix2predcat_arg = bidict()
        arg_preds_per_cat = defaultdict(set)
        for c in self.arg_cats:
            c_ix = self.ix2cat.inverse[c]
            arg_depth = self.cat_arg_depths[c_ix]
            for i in range(arg_depth, max_pred_role_count+1):
                preds_with_i_args = self.preds_by_role_count[i]
                for p in preds_with_i_args:
                    p_ix = self.ix2pred.inv[p]
                    ix2predcat_arg[len(ix2predcat_arg)] = (p_ix, c_ix)
                    arg_preds_per_cat[c_ix].add(p)
        printDebug("ix2predcat_arg:", ix2predcat_arg)

        # maps index of a result predcat to its index in the full set of predcats
        # used during viterbi parsing
        # could also define this for res predcats but it doesn't seem necessary
        argpc_2_pc = [ix2predcat.inv[ix2predcat_arg[i]] for i in range(len(ix2predcat_arg))]
        printDebug("argpc_2_pc:", argpc_2_pc)

        self.pred_counts = torch.tensor(
            [len(preds_per_cat[self.ix2cat.inv[c]]) for c in self.all_cats]
        ).to(self.device)
        # total number of predcats, i.e. (predicate, category) pairs
        self.res_pred_counts = torch.tensor(
            [len(res_preds_per_cat[self.ix2cat.inv[c]]) for c in self.res_cats]
        ).to(self.device)
        self.arg_pred_counts = torch.tensor(
            [len(arg_preds_per_cat[self.ix2cat.inv[c]]) for c in self.arg_cats]
        ).to(self.device)
        self.ix2predcat = ix2predcat
        self.ix2predcat_res = ix2predcat_res
        self.ix2predcat_arg = ix2predcat_arg
        self.argpc_2_pc = torch.tensor(argpc_2_pc).to(self.device)
        self.qall = sum(self.pred_counts).item()
        self.qres = sum(self.res_pred_counts).item()
        self.qarg = sum(self.arg_pred_counts).item()


    def init_association_matrix(self):
        # TODO arg_depth_to_cats needs to be corrected to be different for
        # res and arg (depth-k res cats are not relevant for arg)
        #### define association matrix
        # TODO H_res and H_arg should prolly use indices rather than
        # raw predicates, for consistency with how categories are handled
        self.associations = torch.full(
            (self.qres, self.qarg), fill_value=-QUASI_INF
        )

        # add arg1 associations
        H_res = self.arg1_assoc.keys()
        for h_res in H_res:
            h_res_ix = self.ix2pred.inv[h_res]
            # the result category should have 1 remaining argument for arg1
            # attachment
            C_res = self.arg_depth_to_cats_res[0]
            H_arg = sorted(self.arg1_assoc[h_res].keys())
            C_arg = [self.ix2cat.inv[c] for c in self.arg_cats]
            H_arg_bd = bidict(enumerate(H_arg))
            raw_scores = torch.tensor([self.arg1_assoc[h_res][h] for h in H_arg])
            scores = torch.log_softmax(raw_scores, dim=0)
            for h_arg in H_arg:
                h_arg_ix = self.ix2pred.inv[h_arg]
                score = scores[H_arg_bd.inv[h_arg]]
                for c_res in C_res:
                    for c_arg in C_arg:
                        row = self.ix2predcat_res.inv[(h_res_ix, c_res)]
                        col = self.ix2predcat_arg.inv[(h_arg_ix, c_arg)]
                        assert self.associations[row, col] == -QUASI_INF
                        self.associations[row, col] = score

        # add arg2 associations
        H_res = self.arg2_assoc.keys()
        for h_res in H_res:
            h_res_ix = self.ix2pred.inv[h_res]
            # the result category should have 1 remaining argument for arg2
            # attachment
            C_res = self.arg_depth_to_cats_res[1]
            H_arg = sorted(self.arg2_assoc[h_res].keys())
            C_arg = [self.ix2cat.inv[c] for c in self.arg_cats]
            H_arg_bd = bidict(enumerate(H_arg))
            raw_scores = torch.tensor([self.arg2_assoc[h_res][h] for h in H_arg])
            scores = torch.log_softmax(raw_scores, dim=0)
            for h_arg in H_arg:
                h_arg_ix = self.ix2pred.inv[h_arg]
                score = scores[H_arg_bd.inv[h_arg]]
                for c_res in C_res:
                    for c_arg in C_arg:
                        row = self.ix2predcat_res.inv[(h_res_ix, c_res)]
                        col = self.ix2predcat_arg.inv[(h_arg_ix, c_arg)]
                        assert self.associations[row, col] == -QUASI_INF
                        self.associations[row, col] = score

        torch.set_printoptions(precision=2, linewidth=120)
        printDebug("initial associations:", self.associations)


    def init_masks(self):
        # larg_mask and rarg_mask block impossible argument-result pairs
        # with the argument category on the left and right respectively
        # these only make a difference if categories are read in from
        # a file, not if they're generated by depth
        larg_mask = torch.zeros(
            len(self.res_cats), len(self.arg_cats), dtype=torch.float32
        ).to(self.device)
        rarg_mask = torch.zeros(
            len(self.res_cats), len(self.arg_cats), dtype=torch.float32
        ).to(self.device)

        # TODO move some of this to functor lookup tables
        for res in self.res_cats:
            res_ix = self.ix2cat.inverse[res]
            for arg in self.arg_cats:
                arg_ix = self.ix2cat.inverse[arg]
                # TODO don't hard-code operator
                lfunc = CGNode("-b", res, arg)
                if lfunc not in self.all_cats:
                    rarg_mask[res_ix, arg_ix] = -QUASI_INF
                rfunc = CGNode("-a", res, arg)
                if rfunc not in self.all_cats:
                    larg_mask[res_ix, arg_ix] = -QUASI_INF

        larg_mask = larg_mask.repeat_interleave(
            self.res_pred_counts, dim=0
        ).repeat_interleave(
            self.arg_pred_counts, dim=1
        )
        rarg_mask = rarg_mask.repeat_interleave(
            self.res_pred_counts, dim=0
        ).repeat_interleave(
            self.arg_pred_counts, dim=1
        )
        self.larg_mask = larg_mask.to(self.device)
        self.rarg_mask = rarg_mask.to(self.device)


        root_mask = torch.full(
            (self.qall,), fill_value=-np.inf
        ).to(self.device)
        for ix in range(self.qall):
            pred_ix, cat_ix = self.ix2predcat[ix]
            cat = self.ix2cat[cat_ix]
            if (pred_ix, cat_ix) in self.ix2predcat_res.values() \
                and cat.is_primitive():
                root_mask[ix] = 0

        self.root_mask = root_mask.to(self.device)


    def init_functor_lookup_tables(self):
        # given an result predcat index (i) and an argument predcat 
        # index (j), lfunc_ixs[i, j] gives the functor predcat that
        # takes predcat j as a right argument and returns predcat i
        # e.g. for the rule V -> V-bN N:
        # lfunc_ixs[V, N] = V-bN
        # note that the returned indices are the ones used in ix2predcat,
        # not ix2predcat_res or ix2predcat_arg (which use their own indexing
        # for possible predcats)
        lfunc_ixs = torch.empty(self.qres, self.qarg, dtype=torch.int64)

        # same idea but functor appears on the right
        # e.g. for the rule V -> N V-aN:
        # rfunc_ixs[V, N] = V-aN
        rfunc_ixs = torch.empty(self.qres, self.qarg, dtype=torch.int64)

        for res_pc_ix, (res_p_ix, res_c_ix) in self.ix2predcat_res.items():
            for arg_pc_ix, (_, arg_c_ix) \
                in self.ix2predcat_arg.items():
                res_c = self.ix2cat[res_c_ix]
                arg_c = self.ix2cat[arg_c_ix]
                lfunc_c = CGNode("-b", res_c, arg_c)
                lfunc_c_ix = self.ix2cat.inv[lfunc_c]
                # predicate for functor category is the same as predicate for
                # result category. NOTE: this works for argument attachment
                # but won't work for modifier attachment
                lfunc_p_ix = res_p_ix
                lfunc_pc_ix = self.ix2predcat.inv[(lfunc_p_ix, lfunc_c_ix)]
                lfunc_ixs[res_pc_ix, arg_pc_ix] = lfunc_pc_ix
                rfunc_c = CGNode("-a", res_c, arg_c)
                rfunc_c_ix = self.ix2cat.inv[rfunc_c]
                rfunc_p_ix = res_p_ix
                rfunc_pc_ix = self.ix2predcat.inv[(rfunc_p_ix, rfunc_c_ix)]
                rfunc_ixs[res_pc_ix, arg_pc_ix] = rfunc_pc_ix

        self.lfunc_ixs = lfunc_ixs.to(self.device)
        self.rfunc_ixs = rfunc_ixs.to(self.device)

        
    def forward(self, x, eval=False, argmax=False, use_mean=False, indices=None, set_grammar=True, return_ll=True, **kwargs):
        # x : batch x n
        if set_grammar:
            self.emission = None
            # dim: Qall
#            root_scores = F.log_softmax(
#                self.root_mask+self.root_mlp(self.root_emb).squeeze(), dim=0
#            )
            # all possible root nodes are assigned equal probability.
            # the commented-out alternative assigns root nodes randomly initialized
            # probabilities
            root_scores = F.log_softmax(self.root_mask, dim=0)
            full_p0 = root_scores

            # dim: Qres x 2Qarg
            mlp_out = self.rule_mlp(self.fake_emb)
            if self.arg_depth_penalty:
                mlp_out += self.arg_penalty_mat

            # penalizes rules that use backward function application
            # (in practice encourages right-branching structures)
            if self.left_arg_penalty:
                larg_penalty = torch.full(
                    (self.qres, self.qarg),
                    -self.left_arg_penalty
                )
                rarg_penalty = torch.full((self.qres, self.qarg), 0)
                # dim: Qres x 2Qarg
                penalty = torch.concat(
                    [larg_penalty, rarg_penalty], dim=1
                ).to(self.device)
                mlp_out += penalty

            # dim: Qres x 2Qarg
            rule_scores = F.log_softmax(mlp_out, dim=1)
            # dim: Qres x Qarg
            rule_scores_larg = rule_scores[:, :self.qarg]
            # dim: Qres x Qarg
            rule_scores_rarg = rule_scores[:, self.qarg:]

            nt_emb = self.nt_emb
            # dim: Qfunc x 2
            # split_scores[:, 0] gives P(terminal=0 | cat)
            # split_scores[:, 1] gives P(terminal=1 | cat)
            split_scores = F.log_softmax(self.split_mlp(nt_emb), dim=1)

            # dim: Qres x Qarg
            full_G_larg = rule_scores_larg \
                + split_scores[:self.qres, 0][..., None]
            full_G_rarg = rule_scores_rarg \
                + split_scores[:self.qres, 0][..., None]

            self.parser.set_models(
                full_p0,
                full_G_larg,
                full_G_rarg,
                self.associations,
                self.emission,
                pcfg_split=split_scores
            )


        if self.model_type == 'word':
            x = self.emit_prob_model(x, self.predcat_emb, set_grammar=set_grammar)
        # TODO add predicate embedding to word model
        else:
            assert self.model_type == "char"
            x = self.emit_prob_model(x, self.nt_emb, set_grammar=set_grammar)

        if argmax:
            if eval and self.device != self.eval_device:
                print("Moving model to {}".format(self.eval_device))
                self.parser.device = self.eval_device
            with torch.no_grad():
                logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list = \
                    self.parser.marginal(x, viterbi_flag=True, only_viterbi=not return_ll, sent_indices=indices)
            if eval and self.device != self.eval_device:
                self.parser.device = self.device
                print("Moving model back to {}".format(self.device))
            return logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list
        else:
            logprob_list, _, _, _ = self.parser.marginal(x)
            logprob_list = logprob_list * (-1)
            return logprob_list
