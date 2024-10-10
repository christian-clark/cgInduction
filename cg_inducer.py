import torch, numpy as np, torch.nn.functional as F
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


def readableIx2predCat(ix2predcat, ix2pred, ix2cat):
    readable = bidict()
    for i, (pix, cix) in ix2predcat.items():
        pred = ix2pred[pix]
        cat = ix2cat[cix]
        readable[i] = (pred, cat)
    return readable


class BasicCGInducer(nn.Module):
    def __init__(self, config, num_chars, num_words):
        super(BasicCGInducer, self).__init__()
        self.state_dim = config.getint("state_dim")
        self.rnn_hidden_dim = config.getint("rnn_hidden_dim")
        self.model_type = config["model_type"]
        self.loss_type = config["loss_type"]
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
#        self.arg_depth_penalty = config.getfloat(
#            "arg_depth_penalty", fallback=None
#        )
        self.left_arg_penalty = config.getfloat(
            "left_arg_penalty", fallback=None
        )

        self.init_cats()
        self.init_predicates(config)
        self.init_predcats()
        self.init_association_matrix()
        self.init_masks()
        self.init_functor_lookup_tables()

        # onehot embeddings for predcats
        self.par_predcat_onehot = nn.Parameter(
            torch.eye(self.qpar)
        )
        # onehot embeddings for cats alone
        self.par_cat_onehot = nn.Parameter(
            torch.eye(self.cpar)
        )
        state_dim = self.state_dim
       
        # embeddings for predicate-category pairs
        # used to calculate split scores
        self.all_predcat_emb = nn.Parameter(
            torch.randn(self.qall, state_dim)
        )
        # maps par_cat to gen_cat x {gen_on_L, gen_on_R}
        self.rule_mlp = nn.Linear(
            self.cpar,
            2*self.cgen
        )

        self.root_emb = nn.Parameter(torch.eye(1)).to(self.device)
        self.root_mlp = nn.Linear(1, self.qall).to(self.device)

        # gives P(operation | par_predcat)
        # options for operation: Aa, Ab (preceding and succeeding
        # argument attachment)
        # TODO add Ma and Mb (change 2 to 4)
        self.operation_mlp = nn.Linear(
            self.qpar, 2
        )

        # decides terminal or nonterminal
        self.split_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            nn.Linear(state_dim, 2)
        ).to(self.device)

        self.parser = BatchCKYParser(
            ix2cat=self.ix2cat_all,
            ix2pred=self.ix2pred,
            ix2predcat=self.ix2predcat,
            ix2predcat_par=self.ix2predcat_par,
            ix2predcat_gen=self.ix2predcat_gen,
            genpc_2_pc = self.genpc_2_pc,
            lfunc_ixs=self.lfunc_ixs,
            rfunc_ixs=self.rfunc_ixs,
            larg_mask=self.larg_mask,
            rarg_mask=self.rarg_mask,
            qall=self.qall,
            qpar=self.qpar,
            qgen=self.qgen,
            device=self.device
        )


    def init_cats(self):
        # option 1: specify a set of categories according to maximum depth
        # and number of primitives (used in 2023 ACL Findings paper)
        if self.num_primitives is not None:
            assert self.max_func_depth is not None
            assert self.cats_list is None
            all_cats, par_cats, gen_cats = generate_categories_by_depth(
                self.num_primitives,
                self.max_func_depth,
                self.max_arg_depth
            )

        # option 2: specify a file with a list of categories
        else:
            assert self.cats_list is not None
            # all of this stuff is just for option 1
            assert self.max_func_depth is None
            assert self.max_arg_depth is None
            assert self.left_arg_penalty is None
            all_cats, par_cats, gen_cats = read_categories_from_file(
                self.cats_list
            )
            
        self.all_cats = sorted(all_cats)
        self.call = len(self.all_cats)
        ix2cat_all = bidict()
        for cat in self.all_cats:
            ix2cat_all[len(ix2cat_all)] = cat
        self.ix2cat_all = ix2cat_all
            
        self.par_cats = sorted(par_cats)
        self.cpar = len(self.par_cats)
        ix2cat_par = bidict()
        for cat in self.par_cats:
            ix2cat_par[len(ix2cat_par)] = cat
        self.ix2cat_par = ix2cat_par

        self.gen_cats = sorted(gen_cats)
        self.cgen = len(self.gen_cats)
        ix2cat_gen = bidict()
        for cat in self.gen_cats:
            ix2cat_gen[len(ix2cat_gen)] = cat
        self.ix2cat_gen = ix2cat_gen

        # NOTE: this code is deprecated. To reintroduce a penalty for argument
        # depth, a get_depth method should be added to CGNode in cg_type.py,
        # and then ix2depth should be defined here
#        if self.arg_depth_penalty:
#            printDebug("ix2depth: {}".format(ix2depth))
#            ix2depth_gen = torch.Tensor(ix2depth[:len(gen_cats)])
#            # dim: Qpar x 2Qgen
#            arg_penalty_mat = -self.arg_depth_penalty \
#                * ix2depth_gen.tile((len(par_cats), 2))
#            self.arg_penalty_mat = arg_penalty_mat.to(self.device)

        # cat_arg_depths[i] gives the number of arguments taken by category
        # ix2cat_all[i]
        self.cat_arg_depths = get_category_argument_depths(self.ix2cat_all)
        # arg_depth_to_cats[d] gives the set of categories (indices) with arg depth d
        # used when building association matrix
        self.arg_depth_to_cats_par = defaultdict(set)
        self.arg_depth_to_cats_gen = defaultdict(set)
        for ix, depth in self.cat_arg_depths.items():
            if self.ix2cat_all[ix] in self.par_cats:
                self.arg_depth_to_cats_par[depth].add(ix)
            if self.ix2cat_all[ix] in self.gen_cats:
                self.arg_depth_to_cats_gen[depth].add(ix)


    def init_predicates(self, config):
        predicates = set()
        # pred_role_counts[p] is the number of roles associated with
        # predicate p
        pred_role_counts = defaultdict(int)
        # arg1_assoc[h][j] is the weight for j being h's first argument
        arg1_assoc = defaultdict(dict)
        arg2_assoc = defaultdict(dict)
        # mod_assoc[h][j] is the weight for j modifying h
        mod_assoc = defaultdict(dict)

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
            elif pred1role == 0 and pred2role == 2:
                arg2_assoc[pred2][pred1] = score
            elif pred1role == 0 and pred2role == 0:
                mod_assoc[pred2][pred1] = score
            else:
                raise NotImplementedError(
                    "Only first and second arguments are currently supported. Offending line: {}".format(l)
                )

        preds_by_role_count = defaultdict(set)
        for p in pred_role_counts:
            count = pred_role_counts[p]
            preds_by_role_count[count].add(p)

        ix2pred = bidict(enumerate(sorted(predicates)))
        # use predicate indices instead of raw predicates for
        # indexing associations
        arg1_assoc_ix = defaultdict(dict)
        for h in arg1_assoc:
            hix = ix2pred.inverse[h]
            for j in arg1_assoc[h]:
                jix = ix2pred.inverse[j]
                arg1_assoc_ix[hix][jix] = arg1_assoc[h][j]
        arg2_assoc_ix = defaultdict(dict)
        for h in arg2_assoc:
            hix = ix2pred.inverse[h]
            for j in arg2_assoc[h]:
                jix = ix2pred.inverse[j]
                arg2_assoc_ix[hix][jix] = arg2_assoc[h][j]
        mod_assoc_ix = defaultdict(dict)
        for h in mod_assoc:
            hix = ix2pred.inverse[h]
            for j in arg2_assoc[h]:
                jix = ix2pred.inverse[j]
                mod_assoc_ix[hix][jix] = mod_assoc[h][j]

        self.ix2pred = ix2pred
        self.preds_by_role_count = preds_by_role_count
        self.pred_role_counts = pred_role_counts
        self.arg1_assoc = arg1_assoc_ix
        self.arg2_assoc = arg2_assoc_ix
        self.mod_assoc = mod_assoc_ix


    def init_predcats(self):
        max_pred_role_count = max(self.preds_by_role_count)
        # ix2predcat assigns an index to each valid (predicate, category)
        # pair. E.g., if the index of (DOG, 1/0) is 5, then
        # ix2predcat[5] = (ix2pred.inv[DOG], ix2cat_all.inv[1/0])
        ix2predcat = bidict()
        # preds_per_cat[i] is the set of different predicates that
        # could be assigned to category ix2cat_all[i]
        preds_per_cat = defaultdict(set)
        for c in self.all_cats:
            c_ix = self.ix2cat_all.inv[c]
            arg_depth = self.cat_arg_depths[c_ix]
            for i in range(arg_depth, max_pred_role_count+1):
                preds_with_i_args = self.preds_by_role_count[i]
                for p in preds_with_i_args:
                    p_ix = self.ix2pred.inv[p]
                    ix2predcat[len(ix2predcat)] = (p_ix, c_ix)
                    preds_per_cat[c_ix].add(p)

        ix2predcat_par = bidict()
        par_preds_per_cat = defaultdict(set)
        for c in self.par_cats:
            c_ix = self.ix2cat_all.inverse[c]
            arg_depth = self.cat_arg_depths[c_ix]
            # NOTE: a predicate must have at least n+1 roles to appear with a
            # category of arg depth n. This means that e.g. DOG cannot appear as a
            # parent category even with a primitive category. This rule may need to
            # be changed when modification is added
            for i in range(arg_depth+1, max_pred_role_count+1):
                preds_with_i_args = self.preds_by_role_count[i]
                for p in preds_with_i_args:
                    p_ix = self.ix2pred.inv[p]
                    ix2predcat_par[len(ix2predcat_par)] = (p_ix, c_ix)
                    par_preds_per_cat[c_ix].add(p)

        ix2predcat_gen = bidict()
        gen_preds_per_cat = defaultdict(set)
        for c in self.gen_cats:
            c_ix = self.ix2cat_all.inverse[c]
            arg_depth = self.cat_arg_depths[c_ix]
            for i in range(arg_depth, max_pred_role_count+1):
                preds_with_i_args = self.preds_by_role_count[i]
                for p in preds_with_i_args:
                    p_ix = self.ix2pred.inv[p]
                    ix2predcat_gen[len(ix2predcat_gen)] = (p_ix, c_ix)
                    gen_preds_per_cat[c_ix].add(p)

        # maps index of an gen predcat to its index in the full set of predcats
        # used during viterbi parsing
        genpc_2_pc = [ix2predcat.inv[ix2predcat_gen[i]] for i in range(len(ix2predcat_gen))]
        # maps index of a par predcat to its index in the full set of predcats
        # used to select split scores for binary-branching nodes
        parpc_2_pc = [ix2predcat.inv[ix2predcat_par[i]] for i in range(len(ix2predcat_par))]

        self.pred_counts = torch.tensor(
            [len(preds_per_cat[self.ix2cat_all.inv[c]]) for c in self.all_cats]
        ).to(self.device)
        # total number of predcats, i.e. (predicate, category) pairs
        self.par_pred_counts = torch.tensor(
            [len(par_preds_per_cat[self.ix2cat_all.inv[c]]) for c in self.par_cats]
        ).to(self.device)
        self.gen_pred_counts = torch.tensor(
            [len(gen_preds_per_cat[self.ix2cat_all.inv[c]]) for c in self.gen_cats]
        ).to(self.device)
        self.ix2predcat = ix2predcat
        self.ix2predcat_par = ix2predcat_par
        self.ix2predcat_gen = ix2predcat_gen
        self.genpc_2_pc = torch.tensor(genpc_2_pc).to(self.device)
        self.parpc_2_pc = torch.tensor(parpc_2_pc).to(self.device)
        self.qall = sum(self.pred_counts).item()
        self.qpar = sum(self.par_pred_counts).item()
        self.qgen = sum(self.gen_pred_counts).item()


    def init_association_matrix(self):
        associations = torch.full(
            (self.qpar, self.qgen), fill_value=-QUASI_INF
        )

        # add arg1 associations
        H_par = self.arg1_assoc.keys()
        for h_par in H_par:
            # the parent category should have 0 remaining arguments for arg1
            # attachment
            C_par = self.arg_depth_to_cats_par[0]
            H_gen = sorted(self.arg1_assoc[h_par].keys())
            C_gen = [self.ix2cat_all.inv[c] for c in self.gen_cats]
            H_gen_bd = bidict(enumerate(H_gen))
            raw_scores = torch.tensor([self.arg1_assoc[h_par][h] for h in H_gen])
            scores = torch.log_softmax(raw_scores, dim=0)
            for h_gen in H_gen:
                score = scores[H_gen_bd.inv[h_gen]]
                for c_par in C_par:
                    for c_gen in C_gen:
                        row = self.ix2predcat_par.inv[(h_par, c_par)]
                        col = self.ix2predcat_gen.inv[(h_gen, c_gen)]
                        assert associations[row, col] == -QUASI_INF
                        associations[row, col] = score

        # add arg2 associations
        H_par = self.arg2_assoc.keys()
        for h_par in H_par:
            # the parent category should have 1 remaining argument for arg2
            # attachment
            C_par = self.arg_depth_to_cats_par[1]
            H_gen = sorted(self.arg2_assoc[h_par].keys())
            C_gen = [self.ix2cat_all.inv[c] for c in self.gen_cats]
            H_gen_bd = bidict(enumerate(H_gen))
            raw_scores = torch.tensor([self.arg2_assoc[h_par][h] for h in H_gen])
            scores = torch.log_softmax(raw_scores, dim=0)
            for h_gen in H_gen:
                score = scores[H_gen_bd.inv[h_gen]]
                for c_par in C_par:
                    for c_gen in C_gen:
                        row = self.ix2predcat_par.inv[(h_par, c_par)]
                        col = self.ix2predcat_gen.inv[(h_gen, c_gen)]
                        assert associations[row, col] == -QUASI_INF
                        associations[row, col] = score

        self.associations = associations


    def init_masks(self):
        # larg_mask and rarg_mask block impossible argument-result pairs
        # with the argument category on the left and right respectively
        # these only make a difference if categories are read in from
        # a file, not if they're generated by depth
        larg_mask = torch.zeros(
            len(self.par_cats), len(self.gen_cats), dtype=torch.float32
        ).to(self.device)
        rarg_mask = torch.zeros(
            len(self.par_cats), len(self.gen_cats), dtype=torch.float32
        ).to(self.device)

        # TODO move some of this to functor lookup tables
        for par in self.par_cats:
            par_ix = self.ix2cat_all.inverse[par]
            for gen in self.gen_cats:
                gen_ix = self.ix2cat_all.inverse[gen]
                # TODO don't hard-code operator
                lfunc = CGNode("-b", par, gen)
                if lfunc not in self.all_cats:
                    rarg_mask[par_ix, gen_ix] = -QUASI_INF
                rfunc = CGNode("-a", par, gen)
                if rfunc not in self.all_cats:
                    larg_mask[par_ix, gen_ix] = -QUASI_INF

        larg_mask = larg_mask.repeat_interleave(
            self.par_pred_counts, dim=0
        ).repeat_interleave(
            self.gen_pred_counts, dim=1
        )
        rarg_mask = rarg_mask.repeat_interleave(
            self.par_pred_counts, dim=0
        ).repeat_interleave(
            self.gen_pred_counts, dim=1
        )
        self.larg_mask = larg_mask.to(self.device)
        self.rarg_mask = rarg_mask.to(self.device)


        root_mask = torch.full(
            (self.qall,), fill_value=-np.inf
        ).to(self.device)
        for ix in range(self.qall):
            pred_ix, cat_ix = self.ix2predcat[ix]
            cat = self.ix2cat_all[cat_ix]
            if (pred_ix, cat_ix) in self.ix2predcat_par.values() \
                and cat.is_primitive():
                root_mask[ix] = 0

        self.root_mask = root_mask.to(self.device)


    def init_functor_lookup_tables(self):
        # given an parent/result predcat index (i) and an argument/generated child 
        # predcat index (j), lfunc_ixs[i, j] gives the functor predcat that
        # takes predcat j as a right argument and returns predcat i
        # e.g. for the rule V -> V-bN N:
        # lfunc_ixs[V, N] = V-bN
        # note that the returned indices are the ones used in ix2predcat,
        # not ix2predcat_par or ix2predcat_gen (which use their own indexing
        # for possible predcats)
        lfunc_ixs = torch.empty(self.qpar, self.qgen, dtype=torch.int64)

        # same idea but functor appears on the right
        # e.g. for the rule V -> N V-aN:
        # rfunc_ixs[V, N] = V-aN
        rfunc_ixs = torch.empty(self.qpar, self.qgen, dtype=torch.int64)

        for par_pc_ix, (par_p_ix, par_c_ix) in self.ix2predcat_par.items():
            for gen_pc_ix, (_, gen_c_ix) \
                in self.ix2predcat_gen.items():
                par_c = self.ix2cat_all[par_c_ix]
                gen_c = self.ix2cat_all[gen_c_ix]
                lfunc_c = CGNode("-b", par_c, gen_c)
                lfunc_c_ix = self.ix2cat_all.inv[lfunc_c]
                # predicate for functor category is the same as predicate for
                # parent category. NOTE: this works for argument attachment
                # but won't work for modifier attachment
                lfunc_p_ix = par_p_ix
                lfunc_pc_ix = self.ix2predcat.inv[(lfunc_p_ix, lfunc_c_ix)]
                lfunc_ixs[par_pc_ix, gen_pc_ix] = lfunc_pc_ix
                rfunc_c = CGNode("-a", par_c, gen_c)
                rfunc_c_ix = self.ix2cat_all.inv[rfunc_c]
                rfunc_p_ix = par_p_ix
                rfunc_pc_ix = self.ix2predcat.inv[(rfunc_p_ix, rfunc_c_ix)]
                rfunc_ixs[par_pc_ix, gen_pc_ix] = rfunc_pc_ix

        self.lfunc_ixs = lfunc_ixs.to(self.device)
        self.rfunc_ixs = rfunc_ixs.to(self.device)

        
    def forward(self, x, eval=False, argmax=False, indices=None, set_grammar=True):
        # x : batch x n
        if set_grammar:
            # dim: Qall
#            root_scores = F.log_softmax(
#                self.root_mask+self.root_mlp(self.root_emb).squeeze(), dim=0
#            )
            # all possible root nodes are assigned equal probability.
            # the commented-out alternative assigns root nodes randomly initialized
            # probabilities
            root_scores = F.log_softmax(self.root_mask, dim=0)
            full_p0 = root_scores

            # dim: Qpar x 2 (assuming 2 available operations)
            operation_scores = self.operation_mlp(self.par_predcat_onehot)
            operation_probs = F.log_softmax(operation_scores, dim=1)
            # dim: Qpar
            op_Aa_probs = operation_probs[:, 0]
            # dim: Qpar
            op_Ab_probs = operation_probs[:, 1]

            # dim: Cpar x 2Cgen
            mlp_out = self.rule_mlp(self.par_cat_onehot)
#            if self.arg_depth_penalty:
#                mlp_out += self.arg_penalty_mat

            # penalizes rules that use backward function application
            # (in practice encourages right-branching structures)
            if self.left_arg_penalty:
                larg_penalty = torch.full(
                    (self.cpar, self.cgen),
                    -self.left_arg_penalty
                )
                rarg_penalty = torch.full((self.cpar, self.cgen), 0)
                # dim: Cpar x 2Cgen
                penalty = torch.concat(
                    [larg_penalty, rarg_penalty], dim=1
                ).to(self.device)
                mlp_out += penalty

            # dim: Cpar x Cgen
            # probabilities for binary-branching nodes where preceding
            # argument attachment (Aa) happens
            rule_scores_Aa = mlp_out[:, :self.cgen]
            rule_probs_Aa = F.log_softmax(rule_scores_Aa, dim=1)

            # dim: Cpar x Cgen
            # probabilities for binary-branching nodes where succeeding
            # argument attachment (Ab) happens
            rule_scores_Ab = mlp_out[:, self.cgen:]
            rule_probs_Ab = F.log_softmax(rule_scores_Ab, dim=1)

            # expand rule probabilities from Cpar x Cgen to Qpar x Qgen
            # first expand to Qpar x Cgen...
            cat_ixs_par = [c_ix for _, c_ix in self.ix2predcat_par.values()]
            cats_par = [self.ix2cat_all[ix] for ix in cat_ixs_par]
            cat_ixs_par = [self.ix2cat_par.inv[c] for c in cats_par]
            cat_ixs_row = torch.tensor(cat_ixs_par).to(self.device)
            cat_ixs_row = cat_ixs_row.unsqueeze(dim=1).repeat(1, self.cgen)
            rule_probs_Aa = rule_probs_Aa.gather(dim=0, index=cat_ixs_row)
            rule_probs_Ab = rule_probs_Ab.gather(dim=0, index=cat_ixs_row)

            # ...then expand to Qpar x Qgen
            cat_ixs_gen = [c_ix for _, c_ix in self.ix2predcat_gen.values()]
            cats_gen = [self.ix2cat_all[ix] for ix in cat_ixs_gen]
            cat_ixs_gen = [self.ix2cat_gen.inv[c] for c in cats_gen]
            cat_ixs_col = torch.tensor(cat_ixs_gen).to(self.device)
            cat_ixs_col = cat_ixs_col.unsqueeze(dim=0).repeat(self.qpar, 1)
            rule_probs_Aa = rule_probs_Aa.gather(dim=1, index=cat_ixs_col)
            rule_probs_Ab = rule_probs_Ab.gather(dim=1, index=cat_ixs_col)

            # dim: Qall x 2
            split_scores = self.split_mlp(self.all_predcat_emb)

            # dim: Qall x 2
            # split_scores[:, 0] gives P(terminal=0 | cat)
            # split_scores[:, 1] gives P(terminal=1 | cat)
            split_probs = F.log_softmax(split_scores, dim=1)

            parpc_ix = self.parpc_2_pc.unsqueeze(dim=1).repeat(1, 2)
            # terminal/nonterminal probabilities for par cats (which 
            # are the ones that can undergo terminal expansion)
            # dim: Qpar x 2
            split_probs_par = split_probs.gather(dim=0, index=parpc_ix)
            
            # dim: Qpar x Qgen
            full_G_Aa = split_probs_par[:, 0][..., None] \
                + op_Aa_probs[..., None] \
                + self.associations \
                + rule_probs_Aa
            full_G_Ab = split_probs_par[:, 0][..., None] \
                + op_Ab_probs[..., None] \
                + self.associations \
                + rule_probs_Ab

            self.parser.set_models(
                full_p0,
                full_G_Aa,
                full_G_Ab,
                split_probs
            )

        if self.model_type == 'word':
            x = self.emit_prob_model(x, self.all_predcat_emb, set_grammar=set_grammar)
        else:
            assert self.model_type == "char"
            x = self.emit_prob_model(x, self.all_predcat_emb, set_grammar=set_grammar)

        if argmax:
            if eval and self.device != self.eval_device:
                print("Moving model to {}".format(self.eval_device))
                self.parser.device = self.eval_device
            with torch.no_grad():
                logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list = \
                    self.parser.get_logprobs(
                        x, loss_type=self.loss_type, viterbi_trees=True
                    )
            if eval and self.device != self.eval_device:
                self.parser.device = self.device
                print("Moving model back to {}".format(self.device))

        else:
            logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list = \
                self.parser.get_logprobs(
                    x, loss_type=self.loss_type
                )
            # TODO is it really necessary to do this only for non-argmax?
            logprob_list = logprob_list * (-1)
        return logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list
