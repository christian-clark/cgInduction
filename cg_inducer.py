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

DEBUG = True
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

def add_predcat(p_ix, c_ix, ix2predcat):
    pair = (p_ix, c_ix)
    if pair not in ix2predcat.values():
        ix2predcat[len(ix2predcat)] = pair


class BasicCGInducer(nn.Module):
    def __init__(self, config, word_lexicon, num_chars):
        super(BasicCGInducer, self).__init__()
        self.config = config
        self.word_lexicon = word_lexicon
        self.state_dim = config.getint("state_dim")
        self.rnn_hidden_dim = config.getint("rnn_hidden_dim")
        self.model_type = config["model_type"]
        self.loss_type = config["loss_type"]
        self.device = config["device"]
        self.eval_device = config["eval_device"]
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
        self.init_functor_lookup_tables()

        if self.model_type == 'char':
            self.emit_prob_model = CharProbRNN(
                num_chars,
                state_dim=self.state_dim,
                hidden_size=self.rnn_hidden_dim
            )
        elif self.model_type == 'word':
            self.emit_prob_model = WordProbFCFixVocabCompound(
                len(word_lexicon),
                self.state_dim,
                word_mask=self.word_mask
            )
        else:
            raise ValueError("Model type should be char or word")

        # onehot embeddings for predcats
        self.par_predcat_onehot = nn.Parameter(
            torch.eye(self.qall)
        )
        # onehot embeddings for cats alone
        self.par_cat_onehot = nn.Parameter(
            torch.eye(self.call)
        )
        state_dim = self.state_dim
       
        # embeddings for predicate-category pairs
        # used to calculate split scores
        self.all_predcat_emb = nn.Parameter(
            torch.randn(self.qall, state_dim)
        )
        # maps par_cat to gen_cat
        self.rule_mlp = nn.Linear(self.call, self.cgen)

        self.root_emb = nn.Parameter(torch.eye(1)).to(self.device)
        self.root_mlp = nn.Linear(1, self.qall).to(self.device)

        #self.lr_mlp = nn.Linear(self.qpar, 2)
        # gives P(operation | par_predcat)
        # operation is one of the following:
        # - Aa: preceding argument attachment
        # - Ab: succeeding argument attachment
        # - Ma: preceding modifier attachment
        # - Mb: succeeding modifier attachment
        self.operation_mlp = nn.Linear(self.qall, 4)

        # decides terminal or nonterminal
        self.split_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            nn.Linear(state_dim, 2)
        ).to(self.device)

        # TODO fix the arguments taken by parser
        self.parser = BatchCKYParser(
            ix2cat=self.ix2cat,
            ix2pred=self.ix2pred,
            ix2predcat=self.ix2predcat,
            ix2predcat_gen=self.ix2predcat_gen,
            genpc_2_pc = self.genpc_2_pc,
            lfunc_ixs=self.lfunc_ixs,
            rfunc_ixs=self.rfunc_ixs,
            qall=self.qall,
            qgen=self.qgen,
            device=self.device
        )


    def init_cats(self):
        # option 1: specify a set of categories according to maximum depth
        # and number of primitives (used in 2023 ACL Findings paper)
        if self.num_primitives is not None:
            assert self.max_func_depth is not None
            assert self.cats_list is None
            all_cats, gen_cats, arg_cats, res_cats = \
                generate_categories_by_depth(
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
            all_cats, gen_cats, arg_cats, res_cats = \
                read_categories_from_file(self.cats_list)
            
        self.all_cats = sorted(all_cats)
        self.call = len(self.all_cats)
        ix2cat = bidict()
        for cat in self.all_cats:
            ix2cat[len(ix2cat)] = cat
        self.ix2cat = ix2cat
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


    def init_predicates(self):
        # all predicates: nouns, adjectives, intransitive verbs,
        # transitive verbs
        preds_all = set()
        # predicates that have a first role: adjectives, intransitive
        # verbs, transitive verbs
        preds_role1 = set()
        # arg1_assoc[h][j] is the weight for predicate j being h's first 
        # argument
        arg1_assoc = defaultdict(dict)
        # predicates that have a second role: transitive verbs
        preds_role2 = set()
        # arg2_assoc[h][j] is the weight for predicate j being h's second 
        # argument
        arg2_assoc = defaultdict(dict)
        # mod_assoc[h][j] is the weight for predicate j modifying predicate
        # h. For all h and j, mod_assoc[h][j] = arg1_assoc[j][h]
        mod_assoc = defaultdict(dict)

        f_assoc = open(self.config["predicate_associations"])
        # header
        f_assoc.readline()
        for l in f_assoc:
            pred1, pred1role, pred2, pred2role, score = l.strip().split()
            pred1role = int(pred1role)
            pred2role = int(pred2role)
            assert pred2role == 0, "second predicate is argument or modificand, and its role should be 0"
            score = float(score)
            preds_all.add(pred1)
            preds_all.add(pred2)
            if pred1role == 1:
                preds_role1.add(pred1)
                arg1_assoc[pred1][pred2] = score
                mod_assoc[pred2][pred1] = score
            elif pred1role == 2:
                preds_role2.add(pred1)
                arg2_assoc[pred1][pred2] = score
            else:
                raise Exception("first predicate is functor or modifier, and its role should be 1 or 2")

        ix2pred = bidict(enumerate(sorted(preds_all)))
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

        self.ix2pred = ix2pred
        self.preds_all = preds_all
        self.preds_role1 = preds_role1
        self.preds_role2 = preds_role2
        self.arg1_assoc = arg1_assoc_ix
        self.arg2_assoc = arg2_assoc_ix
        self.mod_assoc = mod_assoc_ix


    def init_predcats(self):
        # ix2predcat assigns an index to each valid (predicate, category)
        # pair. E.g., if the index of (DOG, 1/0) is 5, then
        # ix2predcat[5] = (ix2pred.inv[DOG], ix2cat_all.inv[1/0])
        # note that ix2predcat_gen uses indices from
        # ix2cat_all, NOT ix2cat_gen
        ix2predcat = bidict()
        ix2predcat_gen = bidict()

        # valid predcats are those such that category depth
        # does not exceed the number of roles the predicate has.
        # generated predcats (those in ix2predcat_gen) must either
        # use a category in self.arg_cats or use a modifier category
        # (u-av)
        for c_ix, c in self.ix2cat.items():
            if c.arg_depth() == 0:
                for p in self.preds_all:
                    p_ix = self.ix2pred.inv[p]
                    add_predcat(p_ix, c_ix, ix2predcat)
                    if c in self.arg_cats:
                        add_predcat(p_ix, c_ix, ix2predcat_gen)

            elif c.arg_depth() == 1:
                for p in self.preds_role1:
                    p_ix = self.ix2pred.inv[p]
                    add_predcat(p_ix, c_ix, ix2predcat)
                    if c in self.arg_cats or c.is_modifier():
                        add_predcat(p_ix, c_ix, ix2predcat_gen)
            else:
                assert c.arg_depth() == 2
                for p in self.preds_role2:
                    p_ix = self.ix2pred.inv[p]
                    add_predcat(p_ix, c_ix, ix2predcat)
                    if c in self.arg_cats:
                        add_predcat(p_ix, c_ix, ix2predcat_gen)

        # maps index of an gen predcat to its index in the full set of predcats.
        # used during viterbi parsing
        genpc_2_pc = [ix2predcat.inv[ix2predcat_gen[i]] for i in range(len(ix2predcat_gen))]
        # maps index of a par predcat to its index in the full set of predcats.
        # used to select split scores for binary-branching nodes
        #parpc_2_pc = [ix2predcat_all.inv[ix2predcat_par[i]] for i in range(len(ix2predcat_par))]
        #self.parpc_2_pc = torch.tensor(parpc_2_pc).to(self.device)
        self.ix2predcat = ix2predcat
        self.ix2predcat_gen = ix2predcat_gen
        self.genpc_2_pc = torch.tensor(genpc_2_pc).to(self.device)
        self.qall = len(ix2predcat)
        self.qgen = len(ix2predcat_gen)


    def init_association_matrix(self):
        # calculate P(h_gen | c_par, h_par, op)
        # dim: 4qall x H
        # 4 is for four operations: Aa, Ab, Ma, Mb
        association_scores = torch.zeros((4*self.qall, len(self.preds_all)))
        aa_offset = 0
        ab_offset = 1
        ma_offset = 2
        mb_offset = 3

        for ix_par, (pix_par, cix_par) in self.ix2predcat.items():
            p_par = self.ix2pred[pix_par]
            c_par = self.ix2cat[cix_par]
            aa_ix = 4*ix_par + aa_offset
            ab_ix = 4*ix_par + ab_offset
            ma_ix = 4*ix_par + ma_offset
            mb_ix = 4*ix_par + mb_offset

            # predicate associations allowed via modifier attachment
            for pix_gen, score in self.mod_assoc[pix_par].items():
                #printDebug("p_gen mod:{}".format(p_gen))
                association_scores[ma_ix, pix_gen] = score
                association_scores[mb_ix, pix_gen] = score
            # if modifier attachment happens, forbid p_gen from being
            # a predicate with no role 1 (e.g. DOG)
            for p_gen in self.preds_all - self.preds_role1:
                pix_gen = self.ix2pred.inv[p_gen]
                #printDebug("p_gen nomod:{}".format(p_gen))
                association_scores[ma_ix, pix_gen] = -QUASI_INF
                association_scores[mb_ix, pix_gen] = -QUASI_INF

            # predicate associations allowed via argument attachment
            if c_par in self.res_cats and c_par.is_primitive() \
                and p_par in self.preds_role1:
                for pix_gen, score in self.arg1_assoc[pix_par].items():
                    #printDebug("p_gen arg1:{}".format(p_gen))
                    association_scores[aa_ix, pix_gen] = score
                    association_scores[ab_ix, pix_gen] = score
            if c_par in self.res_cats and c_par.arg_depth() == 1 \
                and p_par in self.preds_role2:
                for pix_gen, score in self.arg2_assoc[pix_par].items():
                    #printDebug("p_gen arg2:{}".format(p_gen))
                    association_scores[aa_ix, pix_gen] = score
                    association_scores[ab_ix, pix_gen] = score

        # dim: 4qall x H
        associations = torch.softmax(association_scores, dim=1)
        # dim: qall x 4 x H
        associations = associations.reshape((self.qall, 4, len(self.preds_all)))
        # associations[i, j, k] is the the probability of parent predcat i
        # generating predicate j under operation k
        # dim: qall x H x 4
        associations = associations.permute((0, 2, 1))
        self.associations = associations.to(self.device)


    def init_masks(self):
        # lgen_mask and rgen_mask block impossible parent-gen pairs
        # with the generated child on the left and right respectively
        # e.g. if 0 is in par_cats and 1 is in gen_cats but 0/1 is not in
        # all cats, then (0, 1) should be blocked as a parent-gen pair
        # these only make a difference if categories are read in from
        # a file, not if they're generated by depth
        lfunc_mask = torch.zeros(
            self.call, self.cgen, dtype=torch.float32
        ).to(self.device)
        rfunc_mask = torch.zeros(
            self.call, self.cgen, dtype=torch.float32
        ).to(self.device)

        for par in self.all_cats:
            par_ix = self.ix2cat.inverse[par]
            for gen in self.gen_cats:
                gen_ix = self.ix2cat_gen.inverse[gen]
                if gen not in self.arg_cats:
                    lfunc_mask[par_ix, gen_ix] = -QUASI_INF
                    rfunc_mask[par_ix, gen_ix] = -QUASI_INF
                else:
                    # TODO don't hard-code operator
                    lfunc = CGNode("-b", par, gen)
                    if lfunc not in self.all_cats:
                        lfunc_mask[par_ix, gen_ix] = -QUASI_INF
                    rfunc = CGNode("-a", par, gen)
                    if rfunc not in self.all_cats:
                        rfunc_mask[par_ix, gen_ix] = -QUASI_INF

        # blocks categories that aren't in form u-av from being
        # used as modifiers
        mod_mask = torch.zeros(
            self.cgen, dtype=torch.float32
        ).to(self.device)

        for gen in self.gen_cats:
            gen_ix = self.ix2cat_gen.inverse[gen]
            if not gen.is_modifier():
                mod_mask[gen_ix] = -QUASI_INF

        # root_mask ensures that only primitives can be at the root node
        root_mask = torch.full(
            (self.qall,), fill_value=-np.inf
        ).to(self.device)
        for ix in range(self.qall):
        #    _, cat_ix = self.ix2predcat[ix]
        #    cat = self.ix2cat[cat_ix]
        #    if cat.is_primitive():
        #        root_mask[ix] = 0
            pred_ix, cat_ix = self.ix2predcat[ix]
            cat = self.ix2cat[cat_ix]
            pred = self.ix2pred[pred_ix]
            if cat.is_primitive() and pred in self.preds_role1:
                root_mask[ix] = 0

        # operation is one of the following:
        # - Aa: preceding argument attachment (index 0)
        # - Ab: succeeding argument attachment (index 1)
        # - Ma: preceding modifier attachment (index 2)
        # - Mb: succeeding modifier attachment (index 3)
        # dim: qall x 4
        operation_mask = torch.zeros(
            self.qall, 4, dtype=torch.float32
        ).to(self.device)
        for pc_ix, (p_ix, c_ix) in self.ix2predcat.items():
            c = self.ix2cat[c_ix]
            p = self.ix2pred[p_ix]
            forbidden = False
            if c not in self.res_cats:
                forbidden = True
            else:
                if c.is_primitive() and p in self.preds_role1:
                    continue
                elif c.arg_depth() == 1 and p in self.preds_role2:
                    continue
                else:
                    forbidden = True
            if forbidden:
                # forbid Aa
                operation_mask[pc_ix, 0] = -QUASI_INF
                # forbid Ab
                operation_mask[pc_ix, 1] = -QUASI_INF

        # TODO use word lexicon and ix2predcat to define mask that will be
        # added to dist in char_coding_models
        if "word_knowledge" in self.config:
            word_mask = torch.zeros(
                self.qall, len(self.word_lexicon)
            ).to(self.device)
            f = open(self.config["word_knowledge"])
            for l in f:
                word, pred = l.strip().split()
                printDebug("word_mask")
                printDebug("word:", word)
                printDebug("pred:", pred)
                word_ix = self.word_lexicon[word]
                pred_ix = self.ix2pred.inv[pred]
                # mask is -QUASI_INF for all predicates except the
                # specified one
                word_mask[:, word_ix] = -QUASI_INF
                for ix, (pix, _) in self.ix2predcat.items():
                    if pix == pred_ix:
                        word_mask[ix, word_ix] = 0
        else:
            word_mask = None

        self.lfunc_mask = lfunc_mask
        self.rfunc_mask = rfunc_mask
        self.mod_mask = mod_mask
        self.root_mask = root_mask
        self.operation_mask = operation_mask
        self.word_mask = word_mask


    def init_functor_lookup_tables(self):
        lfunc_ixs = torch.zeros(self.qall, self.qgen, dtype=torch.int64)
        rfunc_ixs = torch.zeros(self.qall, self.qgen, dtype=torch.int64)

        for par_pc_ix, (par_p_ix, par_c_ix) in self.ix2predcat.items():
            par_c = self.ix2cat[par_c_ix]
            par_p = self.ix2pred[par_p_ix]
            # noun preds can't take arguments
            if par_p not in self.preds_role1: continue
            # depth-1 cats can only take arguments if their predicates have
            # a role 2
            if par_p not in self.preds_role2 and par_c.arg_depth() > 0: continue
            for gen_pc_ix, (_, gen_c_ix) in self.ix2predcat_gen.items():
                gen_c = self.ix2cat[gen_c_ix]
                lfunc_c = CGNode("-b", par_c, gen_c)
                if lfunc_c in self.ix2cat.inv:
                    lfunc_c_ix = self.ix2cat.inv[lfunc_c]
                    lfunc_pc_ix = self.ix2predcat.inv[(par_p_ix, lfunc_c_ix)]
                else:
                    lfunc_pc_ix = 0
                rfunc_c = CGNode("-a", par_c, gen_c)
                if rfunc_c in self.ix2cat.inv:
                    rfunc_c_ix = self.ix2cat.inv[rfunc_c]
                    rfunc_pc_ix = self.ix2predcat.inv[(par_p_ix, rfunc_c_ix)]
                else:
                    rfunc_pc_ix = 0
                lfunc_ixs[par_pc_ix, gen_pc_ix] = lfunc_pc_ix
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

            # dim: Qall x 4 (for 4 possible operations)
            op_scores = self.operation_mlp(self.par_predcat_onehot)
            # block argument attachment when it's impossible
            op_scores += self.operation_mask
            # dim: Qall x 4 (for 4 possible operations)
            op_probs = F.log_softmax(op_scores, dim=1)

            # TODO expand associations from Qall x H x 4 to Qall x Qgen x 4
            pred_ixs_gen = [p_ix for p_ix, _ in self.ix2predcat_gen.values()]
            pred_ixs_col = torch.tensor(pred_ixs_gen).to(self.device)
            pred_ixs_col = pred_ixs_col.unsqueeze(dim=0).unsqueeze(dim=-1).repeat(self.qall, 1, 4)
            associations = self.associations.gather(dim=1, index=pred_ixs_col)

            # dim: Call x Cgen
            mlp_out = self.rule_mlp(self.par_cat_onehot)
#            if self.arg_depth_penaby lty:
#                mlp_out += self.arg_penalty_mat

            # penalizes rules that use backward function application
            # (in practice encourages right-branching structures)
            if self.left_arg_penalty:
                larg_penalty = torch.full(
                    (self.call, self.cgen),
                    -self.left_arg_penalty
                )
                rarg_penalty = torch.full((self.call, self.cgen), 0)
                # dim: Cpar x 2Cgen
                penalty = torch.concat(
                    [larg_penalty, rarg_penalty], dim=1
                ).to(self.device)
                mlp_out += penalty


            # dim: Call x Cgen x 4
            # last dimension is for different operations
            rule_scores = mlp_out.unsqueeze(dim=-1).repeat(1, 1, 4)
            # block impossible parent-gen pairs for preceding argument
            # attachment (Aa, index 0), succeeding argument attachment
            # (Ab, index 1), preceding modifier attachment (Ma, index 2),
            # and succeeding modifier attachment (Mb, index 3)
            rule_scores[..., 0] += self.rfunc_mask
            rule_scores[..., 1] += self.lfunc_mask
            rule_scores[..., 2] += self.mod_mask[None, :]
            rule_scores[..., 3] += self.mod_mask[None, :]
            rule_probs = torch.log_softmax(rule_scores, dim=1)

            # expand rule probabilities from Call x Cgen x 4 to Qall x Qgen x 4
            # first expand to Qall x Cgen x 4...
            cat_ixs_par = [c_ix for _, c_ix in self.ix2predcat.values()]
            cat_ixs_row = torch.tensor(cat_ixs_par).to(self.device)
            cat_ixs_row = cat_ixs_row.unsqueeze(dim=1).unsqueeze(dim=-1).repeat(1, self.cgen, 4)
            rule_probs = rule_probs.gather(dim=0, index=cat_ixs_row)

            # ...then expand to Qall x Qgen x 4
            cat_ixs_gen = [c_ix for _, c_ix in self.ix2predcat_gen.values()]
            # need to reindex according to ix2cat_gen
            cats_gen = [self.ix2cat[ix] for ix in cat_ixs_gen]
            cat_ixs_gen = [self.ix2cat_gen.inv[c] for c in cats_gen]
            cat_ixs_col = torch.tensor(cat_ixs_gen).to(self.device)
            cat_ixs_col = cat_ixs_col.unsqueeze(dim=0).unsqueeze(dim=-1).repeat(self.qall, 1, 4)
            rule_probs = rule_probs.gather(dim=1, index=cat_ixs_col)

            # dim: Qall x 2
            # split_probs[:, 0] gives P(terminal=0 | cat)
            # split_probs[:, 1] gives P(terminal=1 | cat)
            split_scores = self.split_mlp(self.all_predcat_emb)
            split_probs = F.log_softmax(split_scores, dim=1)

            full_G = split_probs[:, 0][..., None, None] \
                + op_probs[:, None, :] \
                + associations \
                + rule_probs

            self.parser.set_models(
                full_p0,
                full_G,
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
