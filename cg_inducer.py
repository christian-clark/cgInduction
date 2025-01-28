from bidict import bidict
from collections import defaultdict
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cky_parser_sgd import BatchCKYParser
from cg_type import CGNode, generate_categories_by_depth, \
    read_categories_from_file, get_category_argument_depths

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


class BasicCGInducer(nn.Module):
    def __init__(self, config, num_words):
        super(BasicCGInducer, self).__init__()
        self.loss_type = config["loss_type"]
        self.device = config["device"]
        self.eval_device = config["eval_device"]
        self.num_primitives = config.getint("num_primitives", fallback=None)
        self.max_func_depth = config.getint("max_func_depth", fallback=None)
        self.max_arg_depth = config.getint("max_arg_depth", fallback=None)
        self.cats_list = config.get("category_list", fallback=None)

        self.init_cats()
        print("CEC cats:", self.all_cats)
        print("CEC cat arg depths:", self.cat_arg_depths)
        print("CEC parcat_2_allcat:", self.parcat_2_allcat)
        self.init_predicates(config)
        print("CEC predicates:", self.predicates)
        print("CEC ix2pred:", self.ix2pred)
        print("CEC valences:", self.valences)
        print("CEC assoc_arg1:", self.assoc_arg1)
        print("CEC assoc_arg2:", self.assoc_arg2)
        
        #self.init_predcats()
        #self.init_association_matrix()
        self.init_masks()
        print("CEC larg_mask:", self.larg_mask)
        print("CEC rarg_mask:", self.rarg_mask)
        print("CEC root_cat_mask:", self.root_cat_mask)
        print("CEC pred_valence_mask:", self.pred_valence_mask)
        self.init_functor_lookup_tables()

        # P(ROOT -> c, p)
        # - only primitive categories can appear at the root
        # Grammar rules
        # P(o | c, p)
        # P(c' | c, o)
        # P(p' | c, p)
        # P(c_left | c, o, c') (deterministic)
        # P(c_right | c, o, c') (deterministic)
        # P(p_left | p, o, p') (deterministic)
        # P(p_right | p, o, p') (deterministic)
        # Lexical rules
        # P(o=lex | c, p)
        # P(w | c, p, o=lex)

        self.root_emb = nn.Parameter(torch.eye(1)).to(self.device)

        # P(c | ROOT)
        self.root_cat_mlp = nn.Linear(1, self.call).to(self.device)
        # P(p | ROOT)
        self.root_pred_mlp = nn.Linear(1, self.p).to(self.device)

        # one-hot embeddings for all (category, predicate) pairs
        self.catpred_emb = nn.Parameter(
            torch.eye(self.call*self.p)
        )

#        # one-hot embeddings for (category, predicate) pairs
#        self.par_catpred_emb = nn.Parameter(
#            torch.eye(self.cpar*self.p)
#        )

        # one-hot embeddings for (category, operation) pairs
        self.par_catop_emb = nn.Parameter(
            torch.eye(self.cpar*2)
        )
#        self.par_cat_emb = nn.Parameter(
#            torch.eye(self.cpar)
#        ).to(self.device)
#        self.par_pred_emb = nn.Parameter(
#            torch.eye(self.p)
#        ).to(self.device)
#
        # P(o | c, p)
        # o \in {Aa, Ab, lex}
        self.op_mlp = nn.Linear(self.call*self.p, 3).to(self.device)

        # P(c' | c, o)
        # o \in {Aa, Ab}
        self.cat_mlp = nn.Linear(self.cpar*2, self.cgen).to(self.device)

        # P(p' | c, p)
        # not learnable; based on association matrix
        # don't need to condition on o because either way it's argument
        # attachment
        #self.assoc_scores = torch.stack([self.assoc_arg1, self.assoc_arg2], dim=0)
        self.assoc_scores = torch.zeros(
            (self.call*self.p, self.p)
        ).to(self.device)
        for c_ix, cat in self.ix2cat_all.items():
            argdepth = cat.arg_depth()
            for p_ix in range(self.p):
                if argdepth == 0:
                    # arg1 association if parent is a primitive
                    assoc = self.assoc_arg1[p_ix]
                elif argdepth == 1:
                    # arg2 association if parent has depth 1
                    assoc = self.assoc_arg2[p_ix]
                else:
                    # otherwise just dummy values
                    assoc = torch.full((self.p,), fill_value=1/self.p)
                row_ix = c_ix*self.p + p_ix
                self.assoc_scores[row_ix] = assoc

        print("CEC assoc_scores:", self.assoc_scores)
        print("CEC assoc_scores shape:", self.assoc_scores.shape)

        # P(w | c, p, o=lex)
        self.word_mlp = nn.Linear(
            self.call*self.p, num_words
        ).to(self.device)

        # TODO fix args
        self.parser = BatchCKYParser(
            ix2cat=self.ix2cat_all,
            ix2cat_par=self.ix2cat_par,
            ix2cat_gen=self.ix2cat_gen,
            ix2pred=self.ix2pred,
            assoc_scores=self.assoc_scores,
            lfunc_ixs=self.lfunc_ixs,
            rfunc_ixs=self.rfunc_ixs,
            larg_mask=self.larg_mask,
            rarg_mask=self.rarg_mask,
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

        parcat_2_allcat = [ix2cat_all.inv[ix2cat_par[i]] for i in ix2cat_par]
        self.parcat_2_allcat = torch.tensor(parcat_2_allcat).to(self.device)

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
        predicates = list()
        valences = list()
        pred_f = config["predicates"]
        for l in open(pred_f):
            pred, valence = l.strip().split(',')
            valence = int(valence)
            predicates.append(pred)
            valences.append(valence)
        ix2pred = bidict(enumerate(predicates))

        assoc_arg1 = list()
        assoc_f = config["associations_arg1"]
        for row in csv.reader(open(assoc_f)):
            assoc = [float(a) for a in row]
            assoc_arg1.append(assoc)
        assoc_arg2 = list()
        assoc_f = config["associations_arg2"]
        for row in csv.reader(open(assoc_f)):
            assoc = [float(a) for a in row]
            assoc_arg2.append(assoc)

        self.predicates = predicates
        self.p = len(self.predicates)
        self.qpar = self.cpar * self.p
        self.qgen = self.cgen * self.p
        self.qall = self.call * self.p
        self.ix2pred = ix2pred
        self.valences = valences
        self.assoc_arg1 = torch.tensor(assoc_arg1)
        self.assoc_arg2 = torch.tensor(assoc_arg2)


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

        self.larg_mask = larg_mask.to(self.device)
        self.rarg_mask = rarg_mask.to(self.device)

        # ensures that only primitive categories are allowed at the root
        root_cat_mask = torch.full(
            (self.call,), fill_value=-QUASI_INF

        )
        for ix, cat in self.ix2cat_all.items():
            if cat.is_primitive():
                root_cat_mask[ix] = 0

        # for a category of depth i, blocks predicates with valence < i
        # dim: call x p
        pred_valence_mask = torch.full(
            (self.call, self.p), fill_value=-QUASI_INF
        )
        for i, cat in self.ix2cat_all.items():
            depth = cat.arg_depth()
            for j in range(self.p):
                valence = self.valences[j]
                if valence >= depth:
                    pred_valence_mask[i, j] = 0

        self.root_cat_mask = root_cat_mask.to(self.device)
        self.pred_valence_mask = pred_valence_mask.to(self.device)

    def init_functor_lookup_tables(self):
        # given a parent/result (cat, pred) index (i) and an argument/
        # generated child (cat, pred) index (j), lfunc_ixs[i, j] gives
        # the functor (cat, pred) index that takes j as a right argument
        # and returns i
        # e.g. for the rule 
        # (1, EAT) -> (1/0, EAT) (0, BUGS)
        # lfunc_ixs[(1, EAT), (0, BUGS)] = (1/0, EAT)
        # note that the returned indices use the category indexing from ix2cat,
        # not ix2cat_par or ix2cat_gen (which use their own indexing
        # for possible cats)
        lfunc_ixs = torch.empty(self.qpar, self.qgen, dtype=torch.int64)

        # same idea but functor appears on the right
        rfunc_ixs = torch.empty(self.qpar, self.qgen, dtype=torch.int64)

        for par_pc_ix in range(self.qpar):
            par_p_ix = par_pc_ix % self.p
            par_c_ix = par_pc_ix // self.p
            par_c = self.ix2cat_par[par_c_ix]
            for gen_pc_ix in range(self.qgen):
                gen_c_ix = gen_pc_ix // self.p
                gen_c = self.ix2cat_gen[gen_c_ix]
                lfunc_c = CGNode("-b", par_c, gen_c)
                lfunc_c_ix = self.ix2cat_all.inv[lfunc_c]
                # functor inherits predicate from parent
                lfunc_pc_ix = lfunc_c_ix*self.p + par_p_ix
                lfunc_ixs[par_pc_ix, gen_pc_ix] = lfunc_pc_ix
                rfunc_c = CGNode("-a", par_c, gen_c)
                rfunc_c_ix = self.ix2cat_all.inv[rfunc_c]
                rfunc_pc_ix = rfunc_c_ix*self.p + par_p_ix
                rfunc_ixs[par_pc_ix, gen_pc_ix] = rfunc_pc_ix
        self.lfunc_ixs = lfunc_ixs.to(self.device)
        self.rfunc_ixs = rfunc_ixs.to(self.device)

        
    def forward(self, x, eval=False, argmax=False, indices=None, set_grammar=True):
        # x : batch x n
        if set_grammar:
            # P(c | ROOT)
            # dim: call
            print("CEC root_emb:", self.root_emb)
            print("CEC unmasked:", self.root_cat_mlp(self.root_emb))
            root_cat_scores = F.log_softmax(
                self.root_cat_mask \
                     + self.root_cat_mlp(self.root_emb).squeeze(),
                dim=0
            )
            print("CEC root_cat_scores:", root_cat_scores)
            # P(p | ROOT)
            # dim: p
            # NOTE: no need to add pred_valence_mask here because the only
            # legal categories are primitives, which can go with any predicate
            root_pred_scores = F.log_softmax(
                self.root_pred_mlp(self.root_emb).squeeze(),
                dim=0
            )
            print("CEC root_pred_scores:", root_pred_scores)

            # P(c, p | ROOT)
            # dim: call x p
            root_scores = root_cat_scores[:, None] + root_pred_scores[None, :]
            # dim: call*p
            root_scores = root_scores.reshape(self.call*self.p)
            print("CEC root_scores:", root_scores)

            # P(o | c, p)
            print("CEC catpred_emb:", self.catpred_emb)
            # this give operation probabilities for all possible cat, pred pairs
            # dim: call*p x 3
            operation_scores = self.op_mlp(self.catpred_emb)
            # TODO mask Aa and Ab operations when predicate's valence isn't
            # high enough
            # dim: call*p x 3
            operation_probs = F.log_softmax(operation_scores, dim=1)
            # dim: call*p
            opLex_probs = operation_probs[:, 2]
            print("CEC operation_probs:", operation_probs)
            # pick out the cpar*p possibilities for binary-branching
            # nodes
            # dim: call x p x 3
            par_operation_probs = operation_probs.reshape(self.call, self.p, 3)
            par_cat_ixs = self.parcat_2_allcat.reshape(self.cpar, 1, 1)
            par_cat_ixs = par_cat_ixs.repeat(1, self.p, 3)
            print("CEC par_cat_ixs:", par_cat_ixs)
            print("CEC gen_cat_ixs shape:", par_cat_ixs.shape)
            # dim: cpar x p x 3
            par_operation_probs = par_operation_probs.gather(index=par_cat_ixs, dim=0)
            # dim: cpar*p x 3
            par_operation_probs = par_operation_probs.reshape(self.cpar*self.p, 3)
            opAa_probs = par_operation_probs[:, 0]
            opAb_probs = par_operation_probs[:, 1]
            print("CEC par_operation_probs:", par_operation_probs)

            # P(c' | c, o)
            # dim: cpar*2 x cgen
            gen_cat_scores = self.cat_mlp(self.par_catop_emb)
            gen_cat_probs = F.log_softmax(gen_cat_scores, dim=1)
            # dim: cpar x 2 x cgen
            gen_cat_probs = gen_cat_probs.reshape(self.cpar, 2, self.cgen)
            print("CEC gen_cat_probs:", gen_cat_probs)

            par_cat_arg_depths = list()
            for cat in self.ix2cat_par.values():
                par_cat_arg_depths.append(cat.arg_depth())
            par_cat_arg_depths = torch.tensor(par_cat_arg_depths)
            print("CEC par_cat_arg_depths:", par_cat_arg_depths)

            # cpar x p x p x 1
            par_cat_arg_depths = par_cat_arg_depths.reshape(-1, 1, 1, 1)
            par_cat_arg_depths = par_cat_arg_depths.repeat(1, self.p, self.p, 1)
            # used to select arg1 or arg2 associations for a parent
            # cat, pred pair
            # dim: cpar*p x p x 1
            par_cat_arg_depths = par_cat_arg_depths.reshape(self.cpar*self.p, self.p, 1)
            print("CEC par_cat_arg_depths expanded:", par_cat_arg_depths)
            print("CEC par_cat_arg_depths expanded shape:", par_cat_arg_depths.shape)

            assoc_stacked = torch.stack([self.assoc_arg1, self.assoc_arg2], dim=2)
            # dim: cpar*p x p x 2
            print("CEC assoc_stacked:", assoc_stacked)
            assoc_stacked = assoc_stacked.repeat(self.cpar, 1, 1)
            print("CEC assoc_stacked repeated:", assoc_stacked)
            print("CEC assoc_stacked shape:", assoc_stacked.shape)

            # P (p' | c, p)
            gen_pred_probs = assoc_stacked.gather(index=par_cat_arg_depths, dim=2)
            # dim: cpar*p x p
            gen_pred_probs = torch.log(gen_pred_probs.squeeze(dim=-1))
            print("CEC gen_pred_probs:", gen_pred_probs)

            # P(c', p', o=Aa | c, p)
            # dim: cpar*p x 1
            op_expand = opAa_probs.unsqueeze(-1)
            # dim: cpar*p x cgen
            cat_expand = gen_cat_probs[:, 0, :].repeat_interleave(self.p, dim=0)
            # dim: cpar*p x cgen*p
            cat_expand = cat_expand.repeat_interleave(self.p, dim=1)
            # dim: cpar*p x cgen*p
            pred_expand = gen_pred_probs.repeat(1, self.cgen)
            # P(c', p', o=Ab | c, p)
            print("CEC op_expand:", op_expand)
            print("CEC op_expand shape:", op_expand.shape)
            print("CEC cat_expand:", cat_expand)
            print("CEC cat_expand shape:", cat_expand.shape)
            print("CEC pred_expand:", pred_expand)
            print("CEC pred_expand shape:", pred_expand.shape)
            full_G_Aa = op_expand + cat_expand + pred_expand
            print("CEC full_G_Aa:", full_G_Aa)

            # P(c', p', o=Ab | c, p)
            # dim: cpar*p x 1
            op_expand = opAb_probs.unsqueeze(-1)
            # dim: cpar*p x cgen
            cat_expand = gen_cat_probs[:, 1, :].repeat_interleave(self.p, dim=0)
            # dim: cpar*p x cgen*p
            cat_expand = cat_expand.repeat_interleave(self.p, dim=1)
            print("CEC op_expand shape:", op_expand.shape)
            print("CEC cat_expand shape:", cat_expand.shape)
            print("CEC pred_expand shape:", pred_expand.shape)
            # P(c', p', o=Ab | c, p)
            full_G_Ab = op_expand + cat_expand + pred_expand
            print("CEC full_G_Ab:", full_G_Ab)

            self.parser.set_models(
                root_scores,
                full_G_Aa,
                full_G_Ab,
                opLex_probs
            )

        # v x c*p
        #dist = F.softmax(self.word_mlp(self.catpred_emb), dim=1).t()
        dist = F.log_softmax(self.word_mlp(self.catpred_emb), dim=1).t()
        # batch x sentlen x c*p
        x = dist[x, :]
        printDebug("CEC x shape:", x.shape)
        #x = self.emit_prob_model(x, self.all_predcat_emb, set_grammar=set_grammar)
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
