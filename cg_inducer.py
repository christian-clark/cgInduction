import torch, numpy as np, torch.nn.functional as F
from bidict import bidict
from collections import defaultdict
from torch import nn
from cky_parser_sgd import BatchCKYParser
from char_coding_models import CharProbRNN, ResidualLayer, \
    WordProbFCFixVocabCompound
from cg_type import CGNode, generate_categories_by_depth, \
    read_categories_from_file

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
        self.config = config
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
        self.init_predicates()
        self.init_predcats()
        self.init_association_matrix()
        self.init_masks()
        self.init_implicit_child_lookup_tables()

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
        # maps par_cat to gen_cat
        self.rule_mlp = nn.Linear(self.cpar, self.cgen)

        self.root_emb = nn.Parameter(torch.eye(1)).to(self.device)
        self.root_mlp = nn.Linear(1, self.qall).to(self.device)

        # gives P(direction | par_predcat)
        # tells whether generated child will be left or right child
        self.lr_mlp = nn.Linear(self.qpar, 2)

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
            ix2predcat_all=self.ix2predcat_all,
            ix2predcat_par=self.ix2predcat_par,
            ix2predcat_gen=self.ix2predcat_gen,
            genpc_2_pc = self.genpc_2_pc,
            limp_ixs=self.limp_ixs,
            rimp_ixs=self.rimp_ixs,
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
            all_cats, par_cats, gen_cats, arg_cats, res_cats = \
                read_categories_from_file(self.cats_list)
            
        self.all_cats = sorted(all_cats)
        self.call = len(self.all_cats)
        ix2cat_all = bidict()
        for cat in self.all_cats:
            ix2cat_all[len(ix2cat_all)] = cat
        self.ix2cat_all = ix2cat_all
        printDebug("ix2cat_all:", self.ix2cat_all)

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

        self.arg_cats = arg_cats
        self.res_cats = res_cats

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
        # shouldn't be needed anymore with arg_depth method in cg_type
#        self.cat_arg_depths = get_category_argument_depths(self.ix2cat_all)

        # arg_depth_to_cats[d] gives the set of categories (indices) with arg depth d
        # used when building association matrix
        self.arg_depth_to_cats_par = defaultdict(set)
        self.arg_depth_to_cats_gen = defaultdict(set)
        #for ix, depth in self.cat_arg_depths.items():
        for ix, cat in self.ix2cat_all.items():
            depth = cat.arg_depth()
            if cat in self.par_cats:
                self.arg_depth_to_cats_par[depth].add(ix)
            if cat in self.gen_cats:
                self.arg_depth_to_cats_gen[depth].add(ix)


    def init_predicates(self):
        predicates = set()
        preds_nounadj = set()
        preds_vtrans = set()
        preds_vintrans = set()
        # pred_role_counts[p] is the number of roles associated with
        # predicate p
        #pred_role_counts = defaultdict(int)
        # arg1_assoc[h][j] is the weight for j being h's first argument
        arg1_assoc = defaultdict(dict)
        arg2_assoc = defaultdict(dict)
        # mod_assoc[h][j] is the weight for j modifying h
        mod_assoc = defaultdict(dict)

        f_assoc = open(self.config["predicate_associations"])
        # header
        f_assoc.readline()
        for l in f_assoc:
            pred1, pred1role, pred2, pred2role, score = l.strip().split()
            pred1role = int(pred1role)
            pred2role = int(pred2role)
            score = float(score)
            predicates.add(pred1)
            predicates.add(pred2)
            #pred_role_counts[pred1] = max(pred1role, pred_role_counts[pred1])
            #pred_role_counts[pred2] = max(pred2role, pred_role_counts[pred2])
            if pred1role == 0 and pred2role == 1:
                # NOTE: reverse the order of the predicates so that the thing
                # taking the argument comes first
                arg1_assoc[pred2][pred1] = score
                preds_nounadj.add(pred1)
                preds_vintrans.add(pred2)
            elif pred1role == 0 and pred2role == 2:
                arg2_assoc[pred2][pred1] = score
                preds_nounadj.add(pred1)
                preds_vtrans.add(pred2)
            elif pred1role == 0 and pred2role == 0:
                mod_assoc[pred2][pred1] = score
                preds_nounadj.add(pred1)
                preds_nounadj.add(pred2)
            else:
                raise NotImplementedError(
                    "Only first and second arguments are currently supported. Offending line: {}".format(l)
                )

        # transitive verbs also may have been added since they have role1
        # as well as role2
        preds_vintrans -= preds_vtrans

#        preds_by_role_count = defaultdict(set)
#        for p in pred_role_counts:
#            count = pred_role_counts[p]
#            preds_by_role_count[count].add(p)

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
            for j in mod_assoc[h]:
                jix = ix2pred.inverse[j]
                mod_assoc_ix[hix][jix] = mod_assoc[h][j]

        printDebug("prelim mod assoc:", mod_assoc_ix)

        # make modifier associations symmetric
        for hix1 in mod_assoc_ix:
            for hix2 in mod_assoc_ix[hix1]:
                mod_assoc_ix[hix2][hix1] = mod_assoc_ix[hix1][hix2]


        printDebug("arg1_assoc:", arg1_assoc_ix)
        printDebug("arg2_assoc:", arg2_assoc_ix)
        printDebug("mod_assoc:", mod_assoc_ix)
        printDebug("predicates:", predicates)
        printDebug("preds_nounadj:", preds_nounadj)
        printDebug("preds_vintrans:", preds_vintrans)
        printDebug("preds_vtrans:", preds_vtrans)


        self.ix2pred = ix2pred
        printDebug("ix2pred:", self.ix2pred)
#        self.preds_by_role_count = preds_by_role_count
#        self.pred_role_counts = pred_role_counts
        self.predicates = predicates
        self.preds_nounadj = preds_nounadj
        self.preds_vintrans = preds_vintrans
        self.preds_vtrans = preds_vtrans
        self.arg1_assoc = arg1_assoc_ix
        self.arg2_assoc = arg2_assoc_ix
        self.mod_assoc = mod_assoc_ix


    def init_predcats(self):
        #max_pred_role_count = max(self.preds_by_role_count)
        # ix2predcat assigns an index to each valid (predicate, category)
        # pair. E.g., if the index of (DOG, 1/0) is 5, then
        # ix2predcat[5] = (ix2pred.inv[DOG], ix2cat_all.inv[1/0])
        ix2predcat_all = bidict()
        ix2predcat_par = bidict()
        ix2predcat_gen = bidict()
        # preds_per_cat[i] is the set of different predicates that
        # could be assigned to category ix2cat_all[i]
        preds_per_cat_all = defaultdict(set)
        preds_per_cat_par = defaultdict(set)
        preds_per_cat_gen = defaultdict(set)

        for c_ix, c in self.ix2cat_all.items():
            if c.is_modifier() or c in self.arg_cats:
                if c.is_modifier():
                    printDebug("modifier cat:", c)
                if c in self.arg_cats:
                    printDebug("arg cat:", c)
                for p in self.preds_nounadj:
                    p_ix = self.ix2pred.inv[p]
                    pair = (p_ix, c_ix)
                    preds_per_cat_all[c_ix].add(p_ix)
                    if pair not in ix2predcat_all.values():
                        ix2predcat_all[len(ix2predcat_all)] = pair
                    preds_per_cat_par[c_ix].add(p_ix)
                    if pair not in ix2predcat_par.values():
                        ix2predcat_par[len(ix2predcat_par)] = pair
                    preds_per_cat_gen[c_ix].add(p_ix)
                    if pair not in ix2predcat_gen.values():
                        ix2predcat_gen[len(ix2predcat_gen)] = pair
            if c in self.res_cats:
                printDebug("res cat:", c)
                if c.arg_depth() == 1:
                    for p in self.preds_vtrans:
                        p_ix = self.ix2pred.inv[p]
                        pair = (p_ix, c_ix)
                        preds_per_cat_all[c_ix].add(p_ix)
                        if pair not in ix2predcat_all.values():
                            ix2predcat_all[len(ix2predcat_all)] = pair
                        preds_per_cat_par[c_ix].add(p_ix)
                        if pair not in ix2predcat_par.values():
                            ix2predcat_par[len(ix2predcat_par)] = pair
                else:
                    assert c.arg_depth() == 0, "c: {}; arg depth: {}".format(c, c.arg_depth())
                    for p in self.preds_vtrans.union(self.preds_vintrans):
                        p_ix = self.ix2pred.inv[p]
                        pair = (p_ix, c_ix)
                        preds_per_cat_all[c_ix].add(p_ix)
                        if pair not in ix2predcat_all.values():
                            ix2predcat_all[len(ix2predcat_all)] = pair
                        preds_per_cat_par[c_ix].add(p_ix)
                        if pair not in ix2predcat_par.values():
                            ix2predcat_par[len(ix2predcat_par)] = pair
            if c.is_primitive():
                printDebug("primitive cat:", c)
                for p in self.preds_nounadj:
                    p_ix = self.ix2pred.inv[p]
                    pair = (p_ix, c_ix)
                    preds_per_cat_all[c_ix].add(p_ix)
                    if pair not in ix2predcat_all.values():
                        ix2predcat_all[len(ix2predcat_all)] = pair
                    preds_per_cat_par[c_ix].add(p_ix)
                    if pair not in ix2predcat_par.values():
                        ix2predcat_par[len(ix2predcat_par)] = pair
                for p in self.preds_vtrans.union(self.preds_vintrans):
                    p_ix = self.ix2pred.inv[p]
                    pair = (p_ix, c_ix)
                    preds_per_cat_all[c_ix].add(p_ix)
                    if pair not in ix2predcat_all.values():
                        ix2predcat_all[len(ix2predcat_all)] = pair
            if c.arg_depth() == 1:
                printDebug("depth 1 cat:", c)
                for p in self.preds_vtrans.union(self.preds_vintrans):
                    p_ix = self.ix2pred.inv[p]
                    pair = (p_ix, c_ix)
                    preds_per_cat_all[c_ix].add(p_ix)
                    if pair not in ix2predcat_all.values():
                        ix2predcat_all[len(ix2predcat_all)] = pair
            if c.arg_depth() == 2:
                printDebug("depth 2 cat:", c)
                for p in self.preds_vtrans:
                    p_ix = self.ix2pred.inv[p]
                    pair = (p_ix, c_ix)
                    preds_per_cat_all[c_ix].add(p_ix)
                    if pair not in ix2predcat_all.values():
                        ix2predcat_all[len(ix2predcat_all)] = pair

        printDebug("ix2predcat_all:", ix2predcat_all)
        printDebug("ix2predcat_par:", ix2predcat_par)
        printDebug("ix2predcat_gen:", ix2predcat_gen)

        printDebug("preds_per_cat_all:", preds_per_cat_all)
        printDebug("preds_per_cat_par:", preds_per_cat_par)
        printDebug("preds_per_cat_gen:", preds_per_cat_gen)

#        for c in self.all_cats:
#            c_ix = self.ix2cat_all.inv[c]
#            arg_depth = self.cat_arg_depths[c_ix]
#            for i in range(arg_depth, max_pred_role_count+1):
#                preds_with_i_args = self.preds_by_role_count[i]
#                for p in preds_with_i_args:
#                    p_ix = self.ix2pred.inv[p]
#                    ix2predcat[len(ix2predcat)] = (p_ix, c_ix)
#                    preds_per_cat[c_ix].add(p)
#
#        #ix2predcat_par = bidict()
#        par_preds_per_cat = defaultdict(set)
#        for c in self.par_cats:
#            c_ix = self.ix2cat_all.inverse[c]
#            if self.config.getboolean("modifier_ops"):
#                min_pred_role_count = self.cat_arg_depths[c_ix]
#            # if only argument attachment is used, a predicate must have at
#            # least n+1 roles to appear with a parent category of arg depth n
#            else:
#                min_pred_role_count = self.cat_arg_depths[c_ix]+1
#            for i in range(min_pred_role_count, max_pred_role_count+1):
#                preds_with_i_args = self.preds_by_role_count[i]
#                for p in preds_with_i_args:
#                    p_ix = self.ix2pred.inv[p]
#                    ix2predcat_par[len(ix2predcat_par)] = (p_ix, c_ix)
#                    par_preds_per_cat[c_ix].add(p)
#
#        ix2predcat_gen = bidict()
#        gen_preds_per_cat = defaultdict(set)
#        for c in self.gen_cats:
#            c_ix = self.ix2cat_all.inverse[c]
#            arg_depth = self.cat_arg_depths[c_ix]
#            for i in range(arg_depth, max_pred_role_count+1):
#                preds_with_i_args = self.preds_by_role_count[i]
#                for p in preds_with_i_args:
#                    p_ix = self.ix2pred.inv[p]
#                    ix2predcat_gen[len(ix2predcat_gen)] = (p_ix, c_ix)
#                    gen_preds_per_cat[c_ix].add(p)

        # maps index of an gen predcat to its index in the full set of predcats.
        # used during viterbi parsing
        genpc_2_pc = [ix2predcat_all.inv[ix2predcat_gen[i]] for i in range(len(ix2predcat_gen))]
        # maps index of a par predcat to its index in the full set of predcats.
        # used to select split scores for binary-branching nodes
        parpc_2_pc = [ix2predcat_all.inv[ix2predcat_par[i]] for i in range(len(ix2predcat_par))]

        self.pred_counts_all = torch.tensor(
            [len(preds_per_cat_all[self.ix2cat_all.inv[c]]) for c in self.all_cats]
        ).to(self.device)
        # total number of predcats, i.e. (predicate, category) pairs
        self.pred_counts_par = torch.tensor(
            [len(preds_per_cat_par[self.ix2cat_all.inv[c]]) for c in self.par_cats]
        ).to(self.device)
        self.pred_counts_gen = torch.tensor(
            [len(preds_per_cat_gen[self.ix2cat_all.inv[c]]) for c in self.gen_cats]
        ).to(self.device)
        self.ix2predcat_all = ix2predcat_all
        self.ix2predcat_par = ix2predcat_par
        self.ix2predcat_gen = ix2predcat_gen
        self.genpc_2_pc = torch.tensor(genpc_2_pc).to(self.device)
        self.parpc_2_pc = torch.tensor(parpc_2_pc).to(self.device)
        self.qall = sum(self.pred_counts_all).item()
        self.qpar = sum(self.pred_counts_par).item()
        self.qgen = sum(self.pred_counts_gen).item()


    def init_association_matrix(self):
        associations = torch.full(
            (self.qpar, self.qgen), fill_value=-QUASI_INF
        )

        for ix_par, (pix_par, cix_par) in self.ix2predcat_par.items():
            assoc_row = torch.full((self.qgen,), fill_value=-QUASI_INF)
            p_par = self.ix2pred[pix_par]
            c_par = self.ix2cat_all[cix_par]
            for ix_gen, (pix_gen, _) in self.ix2predcat_gen.items():
                assert self.ix2pred[pix_gen] in self.preds_nounadj
                # modifier associations
                if p_par in self.preds_nounadj:
                    assoc = self.mod_assoc[pix_par][pix_gen]
                    assoc_row[ix_gen] = assoc

                # associations between intransitive verbs and their arguments
                elif p_par in self.preds_vintrans:
                    assoc = self.arg1_assoc[pix_par][pix_gen]
                    assoc_row[ix_gen] = assoc

                else:
                    assert p_par in self.preds_vtrans
                    # associations between transitive verbs and their first 
                    # arguments
                    if c_par.arg_depth() == 0:
                        assoc = self.arg1_assoc[pix_par][pix_gen]
                        assoc_row[ix_gen] = assoc
                    # associations between transitive verbs and their second
                    # arguments
                    else:
                        assert c_par.arg_depth() == 1
                        assoc = self.arg2_assoc[pix_par][pix_gen]
                        assoc_row[ix_gen] = assoc

            scores = torch.log_softmax(assoc_row, dim=0)
            associations[ix_par] = scores
        self.associations = associations


#        self.arg_associations = arg_associations
#        self.mod_associations = mod_associations

        # =====================
#        arg_associations = torch.full(
#            (self.qpar, self.qgen), fill_value=-QUASI_INF
#        )
#
#        # add arg1 associations
#        H_par = self.arg1_assoc.keys()
#        for h_par in H_par:
#            # the parent category should have 0 remaining arguments for arg1
#            # attachment
#            C_par = self.arg_depth_to_cats_par[0]
#            H_gen = sorted(self.arg1_assoc[h_par].keys())
#            C_gen = [self.ix2cat_all.inv[c] for c in self.gen_cats]
#            H_gen_bd = bidict(enumerate(H_gen))
#            raw_scores = torch.tensor([self.arg1_assoc[h_par][h] for h in H_gen])
#            scores = torch.log_softmax(raw_scores, dim=0)
#            for h_gen in H_gen:
#                score = scores[H_gen_bd.inv[h_gen]]
#                for c_par in C_par:
#                    for c_gen in C_gen:
#                        row = self.ix2predcat_par.inv[(h_par, c_par)]
#                        col = self.ix2predcat_gen.inv[(h_gen, c_gen)]
#                        assert arg_associations[row, col] == -QUASI_INF
#                        arg_associations[row, col] = score
#
#        # add arg2 associations
#        H_par = self.arg2_assoc.keys()
#        for h_par in H_par:
#            # the parent category should have 1 remaining argument for arg2
#            # attachment
#            C_par = self.arg_depth_to_cats_par[1]
#            H_gen = sorted(self.arg2_assoc[h_par].keys())
#            C_gen = [self.ix2cat_all.inv[c] for c in self.gen_cats]
#            H_gen_bd = bidict(enumerate(H_gen))
#            raw_scores = torch.tensor([self.arg2_assoc[h_par][h] for h in H_gen])
#            scores = torch.log_softmax(raw_scores, dim=0)
#            for h_gen in H_gen:
#                score = scores[H_gen_bd.inv[h_gen]]
#                for c_par in C_par:
#                    for c_gen in C_gen:
#                        row = self.ix2predcat_par.inv[(h_par, c_par)]
#                        col = self.ix2predcat_gen.inv[(h_gen, c_gen)]
#                        assert arg_associations[row, col] == -QUASI_INF
#                        arg_associations[row, col] = score
#
#        # TODO
#        if self.config.getboolean("modification_ops"):
#            mod_associations = torch.full(
#                (self.qpar, self.qgen), fill_value=-QUASI_INF
#            )
#            H_par = self.mod_assoc.keys()
#            for h_par in H_par:
#                # the parent category should have 0 remaining arguments for arg1
#                # attachment
#                C_par = self.arg_depth_to_cats_par[0]
#                H_gen = sorted(self.arg1_assoc[h_par].keys())
#                C_gen = [self.ix2cat_all.inv[c] for c in self.gen_cats]
#                H_gen_bd = bidict(enumerate(H_gen))
#                raw_scores = torch.tensor([self.arg1_assoc[h_par][h] for h in H_gen])
#                scores = torch.log_softmax(raw_scores, dim=0)
#                for h_gen in H_gen:
#                    score = scores[H_gen_bd.inv[h_gen]]
#                    for c_par in C_par:
#                        for c_gen in C_gen:
#                            row = self.ix2predcat_par.inv[(h_par, c_par)]
#                            col = self.ix2predcat_gen.inv[(h_gen, c_gen)]
#                            assert arg_associations[row, col] == -QUASI_INF
#                            arg_associations[row, col] = score
#        else:
#            mod_associations = None
#
#        self.arg_associations = arg_associations
#        self.mod_associations = mod_associations


    def init_masks(self):
        # lgen_mask and rgen_mask block impossible parent-gen pairs
        # with the generated child on the left and right respectively
        # e.g. if 0 is in par_cats and 1 is in gen_cats but 0/1 is not in
        # all cats, then (0, 1) should be blocked as a parent-gen pair
        # these only make a difference if categories are read in from
        # a file, not if they're generated by depth
        lgen_mask = torch.zeros(
            len(self.par_cats), len(self.gen_cats), dtype=torch.float32
        ).to(self.device)
        rgen_mask = torch.zeros(
            len(self.par_cats), len(self.gen_cats), dtype=torch.float32
        ).to(self.device)

        for par in self.par_cats:
            par_ix = self.ix2cat_par.inverse[par]
            for gen in self.gen_cats:
                # modifiers can combine with other modifiers or with
                # primitive categories
                if (par.is_primitive() or par.is_modifier()) \
                    and gen.is_modifier():
                    continue
                gen_ix = self.ix2cat_gen.inverse[gen]
                # TODO don't hard-code operator
                lfunc = CGNode("-b", par, gen)
                if lfunc not in self.all_cats:
                    rgen_mask[par_ix, gen_ix] = -QUASI_INF
                rfunc = CGNode("-a", par, gen)
                if rfunc not in self.all_cats:
                    lgen_mask[par_ix, gen_ix] = -QUASI_INF

        self.lgen_mask = lgen_mask
        self.rgen_mask = rgen_mask

        # larg_mask and rarg_mask block impossible argument-result pairs
        # with the argument category on the left and right respectively
        # these only make a difference if categories are read in from
        # a file, not if they're generated by depth
#        larg_mask = torch.zeros(
#            len(self.par_cats), len(self.gen_cats), dtype=torch.float32
#        ).to(self.device)
#        rarg_mask = torch.zeros(
#            len(self.par_cats), len(self.gen_cats), dtype=torch.float32
#        ).to(self.device)
#
#        # TODO move some of this to functor lookup tables
#        for par in self.par_cats:
#            par_ix = self.ix2cat_all.inverse[par]
#            for gen in self.gen_cats:
#                gen_ix = self.ix2cat_all.inverse[gen]
#                # TODO don't hard-code operator
#                lfunc = CGNode("-b", par, gen)
#                if lfunc not in self.all_cats:
#                    rarg_mask[par_ix, gen_ix] = -QUASI_INF
#                rfunc = CGNode("-a", par, gen)
#                if rfunc not in self.all_cats:
#                    larg_mask[par_ix, gen_ix] = -QUASI_INF
#
#        larg_mask = larg_mask.repeat_interleave(
#            self.par_pred_counts, dim=0
#        ).repeat_interleave(
#            self.gen_pred_counts, dim=1
#        )
#        rarg_mask = rarg_mask.repeat_interleave(
#            self.par_pred_counts, dim=0
#        ).repeat_interleave(
#            self.gen_pred_counts, dim=1
#        )
#        self.larg_mask = larg_mask.to(self.device)
#        self.rarg_mask = rarg_mask.to(self.device)


        root_mask = torch.full(
            (self.qall,), fill_value=-np.inf
        ).to(self.device)
        for ix in range(self.qall):
            pred_ix, cat_ix = self.ix2predcat_all[ix]
            cat = self.ix2cat_all[cat_ix]
            if (pred_ix, cat_ix) in self.ix2predcat_par.values() \
                and cat.is_primitive():
                root_mask[ix] = 0

        self.root_mask = root_mask.to(self.device)


    def init_implicit_child_lookup_tables(self):
        # given an parent predcat index i and a generated child 
        # predcat index j, limp_ixs[i, j] gives the implicit child predcat
        # that combines on the left with predcat j to produce predcat i
        # e.g. for the rule CHASE:V -> CHASE:V-bN CAT:N:
        # limp_ixs[CHASE:V, CAT:N] = CHASE:V-bN
        # note that the returned indices are the ones used in ix2predcat,
        # not ix2predcat_par or ix2predcat_gen (which use their own indexing
        # for possible predcats)
        # also note that not all pairs of qpar and qgen are valid; these will
        # need to be filtered out elsewhere in the model
        limp_ixs = torch.zeros(self.qpar, self.qgen, dtype=torch.int64)

        # same idea but implicit child appears on the right
        # e.g. for the rule V -> N V-aN:
        # rimp_ixs[V, N] = V-aN
        rimp_ixs = torch.zeros(self.qpar, self.qgen, dtype=torch.int64)

        for par_pc_ix, (par_p_ix, par_c_ix) in self.ix2predcat_par.items():
            for gen_pc_ix, (_, gen_c_ix) in self.ix2predcat_gen.items():
                par_c = self.ix2cat_all[par_c_ix]
                par_p = self.ix2pred[par_p_ix]
                gen_c = self.ix2cat_all[gen_c_ix]
                # if parent predicate is a noun or adj, only modification can
                # happen. This means the implicit child predcat is the same as
                # the parent predcat
                if par_p in self.preds_nounadj:
                    limp_pc_ix = self.ix2predcat_all.inv[(par_p_ix, par_c_ix)]
                    rimp_pc_ix = self.ix2predcat_all.inv[(par_p_ix, par_c_ix)]
                # otherwise parent predicate is a verb (transitive or
                # intransitive), and only argument attachment can happen
                else:
                    limp_c = CGNode("-b", par_c, gen_c)
                    if limp_c in self.ix2cat_all.inv:
                        limp_c_ix = self.ix2cat_all.inv[limp_c]
                        limp_pc_ix = self.ix2predcat_all.inv[
                            (par_p_ix, limp_c_ix)
                        ]
                    else:
                        limp_pc_ix = 0
                    rimp_c = CGNode("-a", par_c, gen_c)
                    if rimp_c in self.ix2cat_all.inv:
                        rimp_c_ix = self.ix2cat_all.inv[rimp_c]
                        rimp_pc_ix = self.ix2predcat_all.inv[
                            (par_p_ix, rimp_c_ix)
                        ]
                    else:
                        rimp_pc_ix = 0
                limp_ixs[par_pc_ix, gen_pc_ix] = limp_pc_ix
                rimp_ixs[par_pc_ix, gen_pc_ix] = rimp_pc_ix

        self.limp_ixs = limp_ixs.to(self.device)
        self.rimp_ixs = rimp_ixs.to(self.device)


# NOTE replaced by init_implicit_child_lookup_tables
#    def init_functor_lookup_tables(self):
#        # given an parent/result predcat index (i) and an argument/generated child 
#        # predcat index (j), lfunc_ixs[i, j] gives the functor predcat that
#        # takes predcat j as a right argument and returns predcat i
#        # e.g. for the rule V -> V-bN N:
#        # lfunc_ixs[V, N] = V-bN
#        # note that the returned indices are the ones used in ix2predcat,
#        # not ix2predcat_par or ix2predcat_gen (which use their own indexing
#        # for possible predcats)
#        lfunc_ixs = torch.empty(self.qpar, self.qgen, dtype=torch.int64)
#
#        # same idea but functor appears on the right
#        # e.g. for the rule V -> N V-aN:
#        # rfunc_ixs[V, N] = V-aN
#        rfunc_ixs = torch.empty(self.qpar, self.qgen, dtype=torch.int64)
#
#        for par_pc_ix, (par_p_ix, par_c_ix) in self.ix2predcat_par.items():
#            for gen_pc_ix, (_, gen_c_ix) \
#                in self.ix2predcat_gen.items():
#                par_c = self.ix2cat_all[par_c_ix]
#                gen_c = self.ix2cat_all[gen_c_ix]
#                lfunc_c = CGNode("-b", par_c, gen_c)
#                lfunc_c_ix = self.ix2cat_all.inv[lfunc_c]
#                # predicate for functor category is the same as predicate for
#                # parent category. NOTE: this works for argument attachment
#                # but won't work for modifier attachment
#                lfunc_p_ix = par_p_ix
#                lfunc_pc_ix = self.ix2predcat.inv[(lfunc_p_ix, lfunc_c_ix)]
#                lfunc_ixs[par_pc_ix, gen_pc_ix] = lfunc_pc_ix
#                rfunc_c = CGNode("-a", par_c, gen_c)
#                rfunc_c_ix = self.ix2cat_all.inv[rfunc_c]
#                rfunc_p_ix = par_p_ix
#                rfunc_pc_ix = self.ix2predcat.inv[(rfunc_p_ix, rfunc_c_ix)]
#                rfunc_ixs[par_pc_ix, gen_pc_ix] = rfunc_pc_ix
#
#        self.lfunc_ixs = lfunc_ixs.to(self.device)
#        self.rfunc_ixs = rfunc_ixs.to(self.device)

        
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
            lr_scores = self.lr_mlp(self.par_predcat_onehot)
            lr_probs = F.log_softmax(lr_scores, dim=1)

            # dim: Cpar x Cgen
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

            # TODO split rule_probs into rule_probs_lgen and rule_probs_rgen,
            # which are the same except that they have lgen_mask and rgen_mask
            # added in respectively before the softmax
            rule_scores_lgen = mlp_out + self.lgen_mask
            rule_scores_rgen = mlp_out + self.rgen_mask

            # dim: Cpar x Cgen
            # probabilities for binary-branching nodes
            rule_probs_lgen = F.log_softmax(rule_scores_lgen, dim=1)
            rule_probs_rgen = F.log_softmax(rule_scores_rgen, dim=1)

            # expand rule probabilities from Cpar x Cgen to Qpar x Qgen
            # first expand to Qpar x Cgen...
            cat_ixs_par = [c_ix for _, c_ix in self.ix2predcat_par.values()]
            cats_par = [self.ix2cat_all[ix] for ix in cat_ixs_par]
            cat_ixs_par = [self.ix2cat_par.inv[c] for c in cats_par]
            cat_ixs_row = torch.tensor(cat_ixs_par).to(self.device)
            cat_ixs_row = cat_ixs_row.unsqueeze(dim=1).repeat(1, self.cgen)
            rule_probs_lgen = rule_probs_lgen.gather(dim=0, index=cat_ixs_row)
            rule_probs_rgen = rule_probs_rgen.gather(dim=0, index=cat_ixs_row)

            # ...then expand to Qpar x Qgen
            cat_ixs_gen = [c_ix for _, c_ix in self.ix2predcat_gen.values()]
            cats_gen = [self.ix2cat_all[ix] for ix in cat_ixs_gen]
            cat_ixs_gen = [self.ix2cat_gen.inv[c] for c in cats_gen]
            cat_ixs_col = torch.tensor(cat_ixs_gen).to(self.device)
            cat_ixs_col = cat_ixs_col.unsqueeze(dim=0).repeat(self.qpar, 1)
            rule_probs_lgen = rule_probs_lgen.gather(dim=1, index=cat_ixs_col)
            rule_probs_rgen = rule_probs_rgen.gather(dim=1, index=cat_ixs_col)

            # dim: Qall x 2
            # split_probs[:, 0] gives P(terminal=0 | cat)
            # split_probs[:, 1] gives P(terminal=1 | cat)
            split_scores = self.split_mlp(self.all_predcat_emb)
            split_probs = F.log_softmax(split_scores, dim=1)

            parpc_ix = self.parpc_2_pc.unsqueeze(dim=1).repeat(1, 2)
            # terminal/nonterminal probabilities for par cats (which 
            # are the ones that can undergo terminal expansion)
            # dim: Qpar x 2
            split_probs_par = split_probs.gather(dim=0, index=parpc_ix)

            # dim: Qpar x Qgen
            full_G_lgen = split_probs_par[:, 0][..., None] \
                + lr_probs[:, 0][..., None] \
                + self.associations \
                + rule_probs_lgen
            full_G_rgen = split_probs_par[:, 0][..., None] \
                + lr_probs[:, 0][..., None] \
                + self.associations \
                + rule_probs_rgen

#            full_G_Aa = split_probs_par[:, 0][..., None] \
#                + op_Aa_probs[..., None] \
#                + self.arg_associations \
#                + rule_probs_Aa
#            full_G_Ab = split_probs_par[:, 0][..., None] \
#                + op_Ab_probs[..., None] \
#                + self.arg_associations \
#                + rule_probs_Ab

            self.parser.set_models(
                full_p0,
                full_G_lgen,
                full_G_rgen,
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
