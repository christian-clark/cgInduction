# import numpy as np
import torch, logging, datetime
from treenode import Node, nodes_to_tree
import torch.nn.functional as F
import numpy as np


DEBUG = False
def printDebug(*args, **kwargs):
    if DEBUG:
        print("DEBUG: ", end="")
        print(*args, **kwargs)


SMALL_NEGATIVE_NUMBER = -1e8
def logsumexp_multiply(a, b):
    max_a = a.max()
    max_b = b.max()
    res = (a - max_a).exp() @ (b - max_b).exp()
    return res.log() + max_a + max_b


# for dense grammar only! ie D must be -1
class BatchCKYParser:
    def __init__(
        self, ix2cat, predicates, lfunc_ixs, rfunc_ixs, larg_mask, rarg_mask, qall, qres,
        qarg, num_preds, device="cpu"
    ):
        self.D = -1
        # TODO is this correct?
        self.K = qres
        self.lexis = None # Preterminal expansion part of the grammar (this will be dense)
        self.G = None     # Nonterminal expansion part of the grammar (usually be a sparse matrix representation)
        self.p0 = None
        # self.viterbi_chart = np.zeros_like(self.chart, dtype=np.float32)
        self.ix2cat = ix2cat
        self.predicates = predicates
        self.lfunc_ixs = lfunc_ixs
        self.rfunc_ixs = rfunc_ixs
        self.larg_mask = larg_mask
        self.rarg_mask = rarg_mask
        self.Qall = qall
        self.Qres = qres
        self.Qarg = qarg
        self.num_preds = num_preds
        self.this_sent_len = -1
        self.counter = 0
        self.vocab_prob_list = []
        self.finished_vocab = set()
        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    # TODO rename pcfg_split
    def set_models(
            self, p0, expansion_larg, expansion_rarg, associations, 
            operator_ixs, emission, pcfg_split=None
        ):
        self.log_G_larg = expansion_larg
        self.log_G_rarg = expansion_rarg
        self.associations = associations
        self.operator_ixs = operator_ixs
        self.log_p0 = p0
        self.log_lexis = emission
        self.pcfg_split = pcfg_split


    def marginal(self, sents, viterbi_flag=False, only_viterbi=False, sent_indices=None):
        printDebug("computing marginal")
        printDebug("sents shape", sents.shape)
        self.sent_indices = sent_indices

        if not only_viterbi:
            self.compute_inside_logspace(sents)
            # nodes_list, logprob_list = self.sample_tree(sents)
            # nodes_list, logprob_list = self.sample_tree_logspace(sents)
            logprob_list = self.marginal_likelihood_logspace(sents)
        else:
            logprob_list = []
        self.viterbi_sent_indices = self.sent_indices
        self.sent_indices = None
        if viterbi_flag:
            with torch.no_grad():
                vnodes = []
                for sent_index, sent in enumerate(sents):
                    sent = sent.unsqueeze(0)
                    self.sent_indices = [self.viterbi_sent_indices[sent_index],]
                    backchart = self.compute_viterbi_inside(sent)
                    this_vnodes = self.viterbi_backtrack(backchart, sent)
                    vnodes += this_vnodes

        self.this_sent_len = -1

        vtree_list, vproduction_counter_dict_list, vlr_branches_list = [], [], []

        if viterbi_flag:
            for sent_index, sent in enumerate(sents):
                vthis_tree, vproduction_counter_dict, vlr_branches = nodes_to_tree(vnodes[sent_index], sent)
                vtree_list.append(vthis_tree)
                vproduction_counter_dict_list.append(vproduction_counter_dict)
                vlr_branches_list.append(vlr_branches)
        else:
            vtree_list, vproduction_counter_dict_list, vlr_branches_list = [None]*len(sents), [None]*len(sents), \
                                                                 [None]*len(sents)

        self.counter+=1
        return logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list


    def compute_inside_logspace(self, sents):
        try:
            self.this_sent_len = len(sents[0])
        except:
            print(sents)
            raise
        batch_size = len(sents)
        sent_len = self.this_sent_len

        # left chart is the left right triangle of the chart, the top row is the lexical items, and the bottom cell is the
        #  top cell. The right chart is the left chart pushed against the right edge of the chart. The chart is a square.
        self.left_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.Qall*self.num_preds)
        ).float().to(self.device)
        self.right_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.Qall*self.num_preds)
        ).float().to(self.device)
        self.get_lexis_prob(sents, self.left_chart)
        self.right_chart[0] = self.left_chart[0]

        for ij_diff in range(1, sent_len):
            imin = 0
            imax = sent_len - ij_diff
            jmin = ij_diff
            jmax = sent_len
            height = ij_diff

            # a square of the left chart
            # dim: height x imax x batch_size x Qall
            b = self.left_chart[0:height, imin:imax]
            # dim: height x imax x batch_size x Qall
            c = torch.flip(self.right_chart[0:height, jmin:jmax], dims=[0])
            # TODO this can be optimized by taking advantage of the fact that
            # maximally deep categories can only appear on preterminal nodes
            # NOTE: doing the logsumexp here means this takes more memory
            # than the parallel line in compute_viterbi_inside

            # probability of argument i on the left followed by functor j
            # on the right
            # dim: height x imax x batch_size x Qall x Qarg
            children_score_larg = torch.logsumexp(
                b[...,None,:self.Qarg*self.num_preds] + c[...,None], dim=0
            )

            # probability of functor i on the left followed by argument j
            # on the right
            # dim: height x imax x batch_size x Qall x Qarg
            children_score_rarg = torch.logsumexp(
                b[...,None] + c[...,None,:self.Qarg*self.num_preds], dim=0
            )

            # probability that parent category i branches into left argument j
            # and right functor i-aj
            # dim: imax x batch_size x Qres x Qarg
            scores_larg = self.log_G_larg.to(self.device).repeat(
                imax, batch_size, 1, 1
            )
            # probability that parent category i branches into left functor i-bj
            # and right argument j
            # dim:  imax x batch_size x Qres x Qarg
            scores_rarg = self.log_G_rarg.to(self.device).repeat(
                imax, batch_size, 1, 1
            )

            # dim: imax x batch_size x Qres x Qarg
            op_ixs_expanded = self.operator_ixs.repeat_interleave(
                self.num_preds, dim=0
            ).repeat_interleave(
                self.num_preds, dim=1
            ).repeat(imax, batch_size, 1, 1).unsqueeze(dim=-1).to(self.device)

            printDebug("op_ixs_expanded shape", op_ixs_expanded.shape)
            printDebug("op_ixs_expanded", op_ixs_expanded)

            associations_expanded = self.associations.repeat(
                imax, batch_size, self.Qres, self.Qarg, 1
            ).to(self.device)

            printDebug("associations_expanded shape", associations_expanded.shape)
            printDebug("associations_expanded", associations_expanded)

            predicate_scores = torch.gather(
                input=associations_expanded, dim=-1, index=op_ixs_expanded
            ).squeeze(dim=-1)
            printDebug("predicate_scores shape", predicate_scores.shape)
            printDebug("scores_larg shape", scores_larg.shape)

            scores_larg += predicate_scores
            scores_rarg += predicate_scores

#            arg1_scores = self.associations[..., 1]
#
#            scores_pred_arg1 = arg1_scores.to(self.device).repeat(
#                imax, batch_size, self.Qres, self.Qarg
#            )
#            printDebug("scores_pred_arg1 shape:", scores_pred_arg1.shape)
#            # add arg1 scores everywhere except where res category equals
#            # arg category. The mask blocks those locations
#            arg1_mask = torch.ones(scores_pred_arg1.shape).to(self.device)
#            assert self.Qres == self.Qarg
#            for i in range(self.Qres):
#                ixmin = i * self.num_preds
#                ixmax = (i+1) * self.num_preds
#                arg1_mask[..., ixmin:ixmax, ixmin:ixmax] = 0
#
#            #scores_pred_arg1 *= arg1_mask
#
#            scores_larg += scores_pred_arg1
#            scores_rarg += scores_pred_arg1


#            scores_pred_mod = self.mod_scores.to(self.device).repeat(
#                imax, batch_size, 1, 1
#            )
#            printDebug("scores_pred_mod shape:", scores_pred_mod.shape)
#            printDebug("scores_pred_mod:", scores_pred_mod)
#            # NOTE: this assumes that Qres and Qarg use the same indexing
#            # (so category res[i] is the same as category arg[i])
#            # the chunks of scores_larg that are looped over are those where
#            # the res and arg categories are the same, i.e. where a modifier
#            # is attached
#            assert self.Qres == self.Qarg
#            for i in range(self.Qres):
#                ixmin = i * self.num_preds
#                ixmax = (i+1) * self.num_preds
#                scores_larg[..., ixmin:ixmax, ixmin:ixmax] += scores_pred_mod
#                scores_rarg[..., ixmin:ixmax, ixmin:ixmax] += scores_pred_mod

            # dim: Qres x Qarg
            # expand to select the correct predicate, not just category
            rfunc_ixs_expanded = self.rfunc_ixs.repeat_interleave(
                self.num_preds, dim=0
            ).repeat_interleave(
                self.num_preds, dim=1
            )
            rfunc_ixs_expanded *= self.num_preds
            # dim: Qres x Qarg
            pred_ixs = torch.arange(self.num_preds).tile(
                self.Qarg*self.num_preds, self.Qres
            ).t().to(self.device)
            rfunc_ixs_expanded += pred_ixs

            # dim: imax x batch_size x Qres x Qarg
            rfunc_ixs_expanded = rfunc_ixs_expanded.repeat(imax, batch_size, 1, 1)

            # dim: Qres x Qarg
            # expand to select the correct predicate, not just category
            lfunc_ixs_expanded = self.lfunc_ixs.repeat_interleave(
                self.num_preds, dim=0
            ).repeat_interleave(
                self.num_preds, dim=1
            )
            lfunc_ixs_expanded *= self.num_preds
            lfunc_ixs_expanded += pred_ixs

            # dim: imax x batch_size x Qres x Qarg
            lfunc_ixs_expanded = lfunc_ixs_expanded.repeat(imax, batch_size, 1, 1)

            # rfunc_ixs[..., i, j] tells the index of the right-child functor
            # going with parent cat i, larg cat j
            # dim: imax x batch_size x Qres x Qarg
            #rfunc_ixs = self.rfunc_ixs.repeat(imax, batch_size, 1, 1)
            # lfunc_ixs[..., i, j] tells the index of the left-child functor
            # going with parent cat i, rarg cat j
            # dim: imax x batch_size x Qres x Qarg
            #lfunc_ixs = self.lfunc_ixs.repeat(imax, batch_size, 1, 1)

            # rearrange children_score_larg to index by result (parent)
            # and argument rather than functor and argument
            # dim: height x imax x batch_size x Qres x Qarg
            children_score_larg = torch.gather(
                children_score_larg, dim=2, index=rfunc_ixs_expanded
            )
            # block impossible result-argument combinations
            if self.larg_mask is not None:
                children_score_larg += self.larg_mask

            # rearrange children_score_rarg to index by result (parent)
            # and argument rather than functor and argument
            # dim: height x imax x batch_size x Qres x Qarg
            children_score_rarg = torch.gather(
                children_score_rarg, dim=2, index=lfunc_ixs_expanded
            )
            # block impossible result-argument combinations
            if self.rarg_mask is not None:
                children_score_rarg += self.rarg_mask

            # TODO can this be done more space-efficiently?
            # dim: imax x batch_size x Qres x Qarg
            y_larg = scores_larg + children_score_larg
            y_rarg = scores_rarg + children_score_rarg

            # combine left and right arg probabilities
            # dim: imax x batch_size x Qres x Qarg
            y1 = torch.logsumexp(torch.stack([y_larg, y_rarg]), dim=0)
            # marginalize over arg categories
            # dim: imax x batch_size x Qres
            y1 = torch.logsumexp(y1, dim=3)

            # before this, y1 just contains probabilities for the Qres
            # result categories.
            # But left_chart and right_chart maintain all Qall
            # categories, so pad y1 to get it up to that size

            # TODO an alternative to padding here might be to initialize
            # left_chart and right_chart to all -inf
            QUASI_INF = 10000000.
            padding = torch.full(
                (imax, batch_size, (self.Qall-self.Qres)*self.num_preds),
                fill_value=-QUASI_INF
            )
            padding = padding.to(self.device)

            # dim: imax x batch_size x Qall
            y1 = torch.concat([y1, padding], dim=2)
            self.left_chart[height, imin:imax] = y1
            self.right_chart[height, jmin:jmax] = y1
        return


    def marginal_likelihood_logspace(self, sents):
        batch_size = len(sents)
        nodes_list = []

        sent_len = self.this_sent_len
        topnode_pdf = self.left_chart[sent_len-1, 0]

        # draw the top node
        p_topnode = topnode_pdf + self.log_p0
        # norm_term = np.linalg.norm(p_topnode,1)
        logprob_e = torch.logsumexp(p_topnode, dim=1)
        logprobs = logprob_e # / np.log(10)

        return logprobs


    def compute_viterbi_inside(self, sent):
        printDebug("computing viterbi inside")

        self.this_sent_len = sent.shape[1]
        batch_size = 1
        sent_len = self.this_sent_len
        self.left_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.Qall*self.num_preds)
        ).float().to(self.device)
        self.right_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.Qall*self.num_preds)
        ).float().to(self.device)
        backtrack_chart = {}
        self.get_lexis_prob(sent, self.left_chart)
        self.right_chart[0] = self.left_chart[0]

        for ij_diff in range(1, sent_len):
            imin = 0
            imax = sent_len - ij_diff
            jmin = ij_diff
            jmax = sent_len
            height = ij_diff

            # a square of the left chart
            # dim: height x imax x batch_size x Qall
            b = self.left_chart[0:height, imin:imax]
            # dim: height x imax x batch_size x Qall
            c = torch.flip(self.right_chart[0:height, jmin:jmax], dims=[0])

            # probability of argument i on the left followed by functor j
            # on the right
            # dim: height x imax x batch_size x Qall x Qarg
            children_score_larg = b[...,None,:self.Qarg*self.num_preds]+c[...,None]

            # probability of functor i on the left followed by argument j
            # on the right
            # dim: height x imax x batch_size x Qall x Qarg
            children_score_rarg = b[...,None]+c[...,None,:self.Qarg*self.num_preds]

            # probability that parent category i branches into left argument j
            # and right functor i-aj
            # dim: height x imax x batch_size x Qres x Qarg
            scores_larg = self.log_G_larg.to(self.device).repeat(
                height, imax, batch_size, 1, 1
            )
            # probability that parent category i branches into left functor i-bj
            # and right argument j
            # dim: height x imax x batch_size x Qres x Qarg
            scores_rarg = self.log_G_rarg.to(self.device).repeat(
                height, imax, batch_size, 1, 1
            )

            # dim: height x imax x batch_size x Qres x Qarg
            op_ixs_expanded = self.operator_ixs.repeat_interleave(
                self.num_preds, dim=0
            ).repeat_interleave(
                self.num_preds, dim=1
            ).repeat(height, imax, batch_size, 1, 1).unsqueeze(dim=-1).to(self.device)

            associations_expanded = self.associations.repeat(
                height, imax, batch_size, self.Qres, self.Qarg, 1
            ).to(self.device)

            predicate_scores = torch.gather(
                input=associations_expanded, dim=-1, index=op_ixs_expanded
            ).squeeze(dim=-1)
            printDebug("v predicate_scores shape", predicate_scores.shape)
            printDebug("v scores_larg shape", scores_larg.shape)

            scores_larg += predicate_scores
            scores_rarg += predicate_scores
        
#            arg1_scores = self.associations[..., 1]
#            scores_pred_arg1 = arg1_scores.to(self.device).repeat(
#                height, imax, batch_size, self.Qres, self.Qarg
#            )
#            # add arg1 scores everywhere except where res category equals
#            # arg category. The mask blocks those locations
#            arg1_mask = torch.ones(scores_pred_arg1.shape).to(self.device)
#            assert self.Qres == self.Qarg
#            for i in range(self.Qres):
#                ixmin = i * self.num_preds
#                ixmax = (i+1) * self.num_preds
#                arg1_mask[..., ixmin:ixmax, ixmin:ixmax] = 0
#
#            #scores_pred_arg1 *= arg1_mask
#
#            scores_larg += scores_pred_arg1
#            scores_rarg += scores_pred_arg1
#
#            scores_pred_mod = self.mod_scores.to(self.device).repeat(
#                height, imax, batch_size, 1, 1
#            )
#            printDebug("scores_pred_mod viterbi shape:", scores_pred_mod.shape)
#            # NOTE: this assumes that Qres and Qarg use the same indexing
#            # (so category res[i] is the same as category arg[i])
#            # the chunks of scores_larg that are looped over are those where
#            # the res and arg categories are the same, i.e. where a modifier
#            # is attached
#            assert self.Qres == self.Qarg
#            for i in range(self.Qres):
#                ixmin = i * self.num_preds
#                ixmax = (i+1) * self.num_preds
#                scores_larg[..., ixmin:ixmax, ixmin:ixmax] += scores_pred_mod
#                scores_rarg[..., ixmin:ixmax, ixmin:ixmax] += scores_pred_mod
#

            # dim: Qres x Qarg
            # expand to select the correct predicate, not just category
            rfunc_ixs_expanded = self.rfunc_ixs.repeat_interleave(
                self.num_preds, dim=0
            ).repeat_interleave(
                self.num_preds, dim=1
            )
            rfunc_ixs_expanded *= self.num_preds
            # dim: Qres x Qarg
            pred_ixs = torch.arange(self.num_preds).tile(
                self.Qarg*self.num_preds, self.Qres
            ).t().to(self.device)
            rfunc_ixs_expanded += pred_ixs

            # dim: height x imax x batch_size x Qres x Qarg
            rfunc_ixs_expanded = rfunc_ixs_expanded.repeat(height, imax, batch_size, 1, 1)

            # dim: Qres x Qarg
            # expand to select the correct predicate, not just category
            lfunc_ixs_expanded = self.lfunc_ixs.repeat_interleave(
                self.num_preds, dim=0
            ).repeat_interleave(
                self.num_preds, dim=1
            )
            lfunc_ixs_expanded *= self.num_preds
            lfunc_ixs_expanded += pred_ixs

            # dim: height x imax x batch_size x Qres x Qarg
            lfunc_ixs_expanded = lfunc_ixs_expanded.repeat(height, imax, batch_size, 1, 1)

            # rfunc_ixs[..., i, j] tells the index of the right-child functor
            # going with parent cat i, larg cat j
            # dim: height x imax x batch_size x Qres x Qarg
            #rfunc_ixs = self.rfunc_ixs.repeat(height, imax, batch_size, 1, 1)
            # lfunc_ixs[..., i, j] tells the index of the left-child functor
            # going with parent cat i, rarg cat j
            # dim: imax x batch_size x Qres x Qarg
            #lfunc_ixs = self.lfunc_ixs.repeat(height, imax, batch_size, 1, 1)

            # rearrange children_score_larg to index by result (parent)
            # and argument rather than functor and argument
            # dim: height x imax x batch_size x Qres x Qarg
            children_score_larg = torch.gather(
                children_score_larg, dim=3, index=rfunc_ixs_expanded
            )
            # block impossible result-argument combinations
            if self.larg_mask is not None:
                children_score_larg += self.larg_mask

            # rearrange children_score_rarg to index by result (parent)
            # and argument rather than functor and argument
            # dim: height x imax x batch_size x Qres x Qarg
            children_score_rarg = torch.gather(
                children_score_rarg, dim=3, index=lfunc_ixs_expanded
            )
            # block impossible result-argument combinations
            if self.rarg_mask is not None:
                children_score_rarg += self.rarg_mask

            # probability that parent category i branches into left argument j
            # and right functor i-aj, that category j spans the words on the
            # left, and that category i-aj spans the words on the right
            # dim: height x imax x batch_size x Qres x Qarg
            combined_scores_larg = scores_larg + children_score_larg
            # probability that parent category i branches into left functor
            # i-bj and right argument j, that category i-bj spans the words on
            # the left, and that category j spans the words on the right
            # dim: height x imax x batch_size x Qres x Qarg
            combined_scores_rarg = scores_rarg + children_score_rarg

            combined_scores_larg = combined_scores_larg.permute(1,2,3,0,4)
            # dim: imax x batch_size x Qres x height*Qarg
            combined_scores_larg = combined_scores_larg.contiguous().view(
                imax, batch_size, self.Qres*self.num_preds, -1
            )
            combined_scores_rarg = combined_scores_rarg.permute(1,2,3,0,4)
            # dim: imax x batch_size x Qres x height*Qarg
            combined_scores_rarg = combined_scores_rarg.contiguous().view(
                imax, batch_size, self.Qres*self.num_preds, -1
            )

            # dim: imax x batch_size x Qres
            lmax_kbc, largmax_kbc = torch.max(combined_scores_larg, dim=3)
            rmax_kbc, rargmax_kbc = torch.max(combined_scores_rarg, dim=3)

            # dim: imax x batch_size x Qres
            l_ks = torch.div(largmax_kbc, self.Qarg*self.num_preds, rounding_mode="floor") \
                   + torch.arange(1, imax+1)[:, None, None]. to(self.device)
            # dim: imax x batch_size x Qres
            l_bs = largmax_kbc % (self.Qarg*self.num_preds)

            # dim: imax x batch_size x Qres x 1
            l_bs_reshape = l_bs.view(imax, batch_size, self.Qres*self.num_preds, 1)

            # dim: imax x batch_size x Qres x Qarg
#            rfunc_ixs = self.rfunc_ixs.repeat(
#                imax, batch_size, 1, 1
#            )

            l_b_cats = l_bs_reshape // self.Qres

            # dim: imax x batch_size x Qres
            #l_cs = torch.gather(rfunc_ixs, index=l_bs_reshape, dim=3).squeeze(dim=3)
            rfunc_ixs = rfunc_ixs_expanded[0]
            l_cs = torch.gather(rfunc_ixs, index=l_bs_reshape, dim=3).squeeze(dim=3)
 
            # dim: 3 x imax x batch_size x res
            l_kbc = torch.stack([l_ks, l_bs, l_cs], dim=0)

            # dim: imax x batch_size x Qres
            r_ks = torch.div(rargmax_kbc, self.Qarg*self.num_preds, rounding_mode="floor") \
                   + torch.arange(1, imax+1)[:, None, None]. to(self.device)
            # dim: imax x batch_size x Qres
            r_cs = rargmax_kbc % (self.Qarg*self.num_preds)

            # dim: imax x batch_size x Qres x 1
            r_cs_reshape = r_cs.view(imax, batch_size, self.Qres*self.num_preds, 1)

            # dim: imax x batch_size x Qres x Qarg
#            lfunc_ixs = self.lfunc_ixs.repeat(
#                imax, batch_size, 1, 1
#            )

            # dim: imax x batch_size x Qres
            #r_bs = torch.gather(lfunc_ixs, index=r_cs_reshape, dim=3).squeeze(dim=3)
            lfunc_ixs = lfunc_ixs_expanded[0]
            r_bs = torch.gather(lfunc_ixs, index=r_cs_reshape, dim=3).squeeze(dim=3)

            # dim: 3 x imax x batch_size x Qres
            r_kbc = torch.stack([r_ks, r_bs, r_cs], dim=0)

            # dim: 2 x 3 x imax x batch_size x Qres
            lr_kbc = torch.stack([l_kbc, r_kbc], dim=0)

            # dim: 2 x imax x batch_size x Qres
            lr_max = torch.stack([lmax_kbc, rmax_kbc], dim=0)

            # tells whether left arg or right arg is more likely
            # each value of the argmax is 0 (left) or 1 (right)
            # dim: imax x batch_size x Qres
            combined_max, combined_argmax = torch.max(lr_max, dim=0)

            # TODO an alternative to padding here might be to initialize
            # left_chart and right_chart to all -inf
            QUASI_INF = 10000000.
            padding = torch.full(
                (imax, batch_size, (self.Qall-self.Qres)*self.num_preds),
                fill_value=-QUASI_INF
            )
            padding = padding.to(self.device)

            # dim: imax x batch_size x Qall
            combined_max = torch.concat([combined_max, padding], dim=2)

            self.left_chart[height, imin:imax] = combined_max
            self.right_chart[height, jmin:jmax] = combined_max

            # gather k, b, and c
            # dim: 1 x 3 x imax x batch_size x Qres
            combined_argmax = combined_argmax.repeat(3, 1, 1, 1).unsqueeze(dim=0)
            # TODO use different arrangement of dimensions initally to
            # avoid need for permute
            # dim: 3 x imax x batch_size x Qres
            best_kbc = torch.gather(lr_kbc, index=combined_argmax, dim=0).squeeze(dim=0)
            # dim: imax x batch_size x Qres x 3
            best_kbc = best_kbc.permute(1, 2, 3, 0)
            backtrack_chart[ij_diff] = best_kbc
        self.right_chart = None
        return backtrack_chart


    # TODO sent doesn't seem to be a necessary arg
    def viterbi_backtrack(self, backtrack_chart, sent, max_cats=None):
        sent_index = 0
        nodes_list = []
        sent_len = self.this_sent_len
        topnode_pdf = self.left_chart[sent_len-1, 0]
        if max_cats is not None:
            max_cats = max_cats.squeeze()
            max_cats = max_cats.tolist()

        # draw the top node
        p_topnode = topnode_pdf + self.log_p0
        A_ll, top_A = torch.max(p_topnode, dim=-1)
        # top_A = top_A.squeeze()
        # A_ll = A_ll.squeeze()

        expanding_nodes = []
        expanded_nodes = []
        # rules = []
        assert self.this_sent_len > 0, "must call inside pass first!"

        a = top_A[sent_index].item()
        a_cat = a // self.num_preds
        a_pred = a % self.num_preds
        #A_cat_str = str(self.ix2cat[A_cat])
        a_str = "{}:{}".format(self.ix2cat[a_cat], self.predicates[a_pred])
        
        assert not ( torch.isnan(A_ll[sent_index]) or torch.isinf(A_ll[sent_index]) or A_ll[sent_index].item() == 0 ), \
            'something wrong with viterbi parsing. {}'.format(A_ll[sent_index])

        # prepare the downward sampling pass
        top_node = Node(a, a_str, 0, sent_len, self.D, self.K)
        if sent_len > 1:
            expanding_nodes.append(top_node)
        else:
            expanded_nodes.append(top_node)
        # rules.append(Rule(None, A_cat))
        # print(backtrack_chart)
        while expanding_nodes:
            working_node = expanding_nodes.pop()
            ij_diff = working_node.j - working_node.i - 1
            k_b_c = backtrack_chart[ij_diff][ working_node.i, sent_index,
                                                        working_node.cat]
            split_point, b, c = k_b_c[0].item(), k_b_c[1].item(), k_b_c[2].item()
            b_cat = b // self.num_preds
            b_pred = b % self.num_preds
            c_cat = c // self.num_preds
            c_pred = c % self.num_preds
            #b_str = str(self.ix2cat[b])
            b_str = "{}:{}".format(self.ix2cat[b_cat], self.predicates[b_pred])
            #c_str = str(self.ix2cat[c])
            c_str = "{}:{}".format(self.ix2cat[c_cat], self.predicates[c_pred])

            expanded_nodes.append(working_node)
            node_b = Node(b, b_str, working_node.i, split_point, self.D, self.K, parent=working_node)
            node_c = Node(c, c_str, split_point, working_node.j, self.D, self.K, parent=working_node)
            # print(node_b, node_c)
            if node_b.d == self.D and node_b.j - node_b.i != 1:
                print(node_b)
                raise Exception
            if node_b.s != 0 and node_c.s != 1:
                raise Exception("{}, {}".format(node_b, node_c))
            if node_b.is_terminal():
                if max_cats is not None:
                    node_b.k = str(node_b.k) + '|' + str(max_cats[node_b.i][node_b.k])
                expanded_nodes.append(node_b)
                # rules.append(Rule(node_b.k, sent[working_node.i]))
            else:
                expanding_nodes.append(node_b)
            if node_c.is_terminal():
                if max_cats is not None:
                    node_c.k = str(node_c.k) + '|' + str(max_cats[node_c.i][node_c.k])
                expanded_nodes.append(node_c)
                # rules.append(Rule(node_c.k, sent[k]))
            else:
                expanding_nodes.append(node_c)
            # rules.append(Rule(working_node.cat, node_b.k, node_c.k))
        nodes_list.append(expanded_nodes)
        return nodes_list


    def get_lexis_prob(self, sent_embs, left_chart):
        sent_embs = sent_embs.transpose(1, 0) # sentlen, batch, emb
        if isinstance(self.log_lexis, torch.distributions.Distribution):
            sent_embs = sent_embs.unsqueeze(-2) # sentlen, batch, 1, emb
            lexis_probs = self.log_lexis.log_prob(sent_embs) # sentlen, batch, terms

        elif self.log_lexis is None:
            lexis_probs = sent_embs  # sentlen, batch, terms

        else:
            lexis_probs = self.log_lexis.log_prob(sent_embs) # sentlen, batch, terms
        # print('lexical', lexis_probs)
        if self.pcfg_split is not None:
            # Pr(Term=1) only depends on category, not predicate
            pcfg_split_expanded = torch.repeat_interleave(
                self.pcfg_split[:, 1], self.num_preds
            )
            lexis_probs = lexis_probs + pcfg_split_expanded # sentlen, batch, p
            full_lexis_probs = lexis_probs
        else:
            full_lexis_probs = torch.full((lexis_probs.shape[0], lexis_probs.shape[1], self.K), SMALL_NEGATIVE_NUMBER,
                                          device=lexis_probs.device)
            full_lexis_probs = torch.cat([full_lexis_probs, lexis_probs], dim=2)
            print(full_lexis_probs[0,0])
        left_chart[0] = full_lexis_probs
        return

