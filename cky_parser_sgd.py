# import numpy as np
import torch, logging, datetime
from treenode import Node, nodes_to_tree
import torch.nn.functional as F
import numpy as np

QUASI_INF = 10000000.

DEBUG = True
def printDebug(*args, **kwargs):
    if DEBUG:
        print("DEBUG: ", end="")
        print(*args, **kwargs)


def logsumexp_multiply(a, b):
    max_a = a.max()
    max_b = b.max()
    res = (a - max_a).exp() @ (b - max_b).exp()
    return res.log() + max_a + max_b


class BatchCKYParser:
    def __init__(
        self, ix2cat, ix2cat_par, ix2cat_gen, ix2pred, assoc_scores,
        lfunc_ixs, rfunc_ixs, larg_mask, rarg_mask, device="cpu"
    ):
        # TODO figure out/document what D and K do
        # they are used in viterbi_backtrack
        self.D = -1
        self.K = len(ix2cat_par)

        self.ix2cat = ix2cat
        self.ix2cat_par = ix2cat_par
        self.ix2cat_gen = ix2cat_gen
        self.ix2pred = ix2pred
        # number of predicates
        self.p = len(self.ix2pred)
        self.assoc_scores = assoc_scores
        self.lfunc_ixs = lfunc_ixs
        self.rfunc_ixs = rfunc_ixs
        self.larg_mask = larg_mask
        self.rarg_mask = rarg_mask
        # total number of categories
        self.call = len(self.ix2cat)
        # total number of parent categories
        self.cpar = len(self.ix2cat_par)
        # total number of generated categories
        self.cgen = len(self.ix2cat_gen)
        self.qall = self.call * self.p
        self.qpar = self.cpar * self.p
        self.qgen = self.cgen * self.p
        self.this_sent_len = -1
        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'


    def set_models(
            self, root_scores, expansion_Aa, expansion_Ab, opLex_probs
        ):
        self.root_scores = root_scores
        self.log_G_Aa = expansion_Aa
        self.log_G_Ab = expansion_Ab
        self.opLex_probs = opLex_probs

    
    def get_logprobs(
            self, sents, loss_type="marginal", viterbi_trees=False
        ):
        # TODO maybe the two compute_inside_ methods can be merged?
        if loss_type == "marginal":
            left_chart, backtrack_chart = self.compute_inside_marginal(sents)
        else:
            assert loss_type == "best_parse"
            left_chart, backtrack_chart = self.compute_inside_bestparse(sents)
    
        logprob_list = self.likelihood_from_chart(left_chart, loss_type)
        if viterbi_trees:
            vtree_list = list()
            vproduction_counter_dict_list = list()
            vlr_branches_list = list()
            with torch.no_grad():
                # compute_inside_marginal() doesn't produce a backtrack chart
                if backtrack_chart is None:
                    left_chart, backtrack_chart = self.compute_inside_bestparse(sents)
                printDebug("left chart used for viterbi backtracking")
                printDebug(left_chart)
                for sent_index, sent in enumerate(sents):
                    this_vnodes = self.viterbi_backtrack(left_chart, backtrack_chart, sent_index)
                    printDebug("this_vnodes", this_vnodes)
                    vthis_tree, vproduction_counter_dict, vlr_branches = nodes_to_tree(this_vnodes, sent)
                    vtree_list.append(vthis_tree)
                    vproduction_counter_dict_list.append(vproduction_counter_dict)
                    vlr_branches_list.append(vlr_branches)
        else:
            vtree_list = None
            vproduction_counter_dict_list = None
            vlr_branches_list = None

        return logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list


    def compute_inside_marginal(self, sents):
        try:
            self.this_sent_len = len(sents[0])
        except:
            print(sents)
            raise
        batch_size = len(sents)
        sent_len = self.this_sent_len

        # left chart is the left right triangle of the chart, the top row is the lexical items, and the bottom cell is the
        #  top cell. The right chart is the left chart pushed against the right edge of the chart. The chart is a square.
        left_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.qall)
        ).float().to(self.device)
        right_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.qall)
        ).float().to(self.device)
        self.set_lexical_prob(sents, left_chart)
        right_chart[0] = left_chart[0]

        for ij_diff in range(1, sent_len):
            imin = 0
            imax = sent_len - ij_diff
            jmin = ij_diff
            jmax = sent_len
            height = ij_diff

            # a square of the left chart
            # dim: height x imax x batch_size x qall
            b = left_chart[0:height, imin:imax]
            # dim: height x imax x batch_size x call x p
            #b = b.reshape(height, imax, batch_size, self.call, self.p)
            printDebug("b shape:", b.shape)
            # dim: height x imax x batch_size x qall
            c = torch.flip(right_chart[0:height, jmin:jmax], dims=[0])
            #c = c.reshape(height, imax, batch_size, self.call, self.p)
            printDebug("c shape:", c.shape)
            
            # indices for categories that can be generated
            gen_ixs = torch.tensor(
                [self.ix2cat.inv[c] for c in self.ix2cat_gen.values()]
            ).to(self.device)
            # dim: height x imax x batch_size x cgen x p
            gen_ixs = gen_ixs.unsqueeze(-1).repeat(
                height, imax, batch_size, 1, self.p
            )

            gen_ixs = list()
            for gen_pc_ix in range(self.qgen):
                gen_c_ix = gen_pc_ix // self.p
                p_ix = gen_pc_ix % self.p
                all_c_ix = self.ix2cat.inv[self.ix2cat_gen[gen_c_ix]]
                all_pc_ix = all_c_ix*self.p + p_ix
                gen_ixs.append(all_pc_ix)
            # dim: qgen
            gen_ixs = torch.tensor(gen_ixs).to(self.device)
            # dim: height x imax x batch_size x qgen
            gen_ixs = gen_ixs.repeat(height, imax, batch_size, 1)
            printDebug("gen_ixs shape:", gen_ixs.shape)

            # gather relevant categories paired with all possible predicates
            # torch throws an error about inplace modification if the clone()
            # isn't there...idk why
            # dim: height x imax x batch_size x qgen
            b_gen = b.clone().gather(dim=-1, index=gen_ixs)
            printDebug("b_gen shape:", b_gen.shape)
            # probability of argument i on the left followed by functor j
            # on the right
            # dim: imax x batch_size x qall x qgen
            children_score_lgen = torch.logsumexp(
                b_gen[...,None,:] + c[...,None], dim=0
            )
            printDebug("children_score_lgen shape:", children_score_lgen.shape)

            # dim: height x imax x batch_size x qgen
            c_gen = c.gather(dim=-1, index=gen_ixs)
            printDebug("c_gen shape:", c_gen.shape)
            # probability of functor i on the left followed by argument j
            # on the right
            # dim: imax x batch_size x qall x qgen
            children_score_rgen = torch.logsumexp(
                b[...,None] + c_gen[...,None,:], dim=0
            )
            printDebug("children_score_rgen shape:", children_score_lgen.shape)

            # probability that parent category i branches into left argument j
            # and right functor i-aj
            # dim: imax x batch_size x qpar x qgen
            scores_larg = self.log_G_Aa.to(self.device).repeat(
                imax, batch_size, 1, 1
            )
            # probability that parent category i branches into left functor i-bj
            # and right argument j
            # dim:  imax x batch_size x qpar x qgen
            scores_rarg = self.log_G_Ab.to(self.device).repeat(
                imax, batch_size, 1, 1
            )
#            # dim: imax x batch_size x cpar x p x cgen x p
#            scores_rarg = scores_rarg.reshape(
#                imax, batch_size, self.cpar, self.p, self.cgen, self.p
#            )
#            printDebug("scores_rarg shape:", scores_rarg.shape)

            # dim: imax x batch_size x qpar x qgen
            rfunc_ixs = self.rfunc_ixs.repeat(imax, batch_size, 1, 1)
#            rfunc_ixs = rfunc_ixs.unsqueeze(dim=3).unsqueeze(dim=-1)
#            # dim: imax x batch_size x cpar x p x cgen x p
#            rfunc_ixs = rfunc_ixs.repeat(1, 1, 1, self.p, 1, self.p)
#            printDebug("rfunc_ixs shape:", rfunc_ixs.shape)

            # dim: imax x batch_size x qpar x qgen
            lfunc_ixs = self.lfunc_ixs.repeat(imax, batch_size, 1, 1)
#            lfunc_ixs = lfunc_ixs.unsqueeze(dim=3).unsqueeze(dim=-1)
#            # dim: imax x batch_size x cpar x p x cgen x p
#            lfunc_ixs = lfunc_ixs.repeat(1, 1, 1, self.p, 1, self.p)
#            printDebug("lfunc_ixs shape:", lfunc_ixs.shape)

            # rearrange children_score_lgen to index by parent
            # and argument rather than functor and argument
            # dim: imax x batch_size x cpar x p x cgen x p
            children_score_larg = torch.gather(
                children_score_lgen, dim=2, index=rfunc_ixs
            )
            printDebug("children_score_larg shape:", children_score_larg.shape)
            # block impossible parent-argument combinations
            # original dim of larg_mask: cpar x cgen
            larg_mask = self.larg_mask.unsqueeze(1).unsqueeze(-1)
            # dim: cpar x p x cgen x p
            larg_mask = larg_mask.repeat(1, self.p, 1, self.p)
            # dim: qpar x qgen
            larg_mask = larg_mask.reshape(self.qpar, self.qgen)
            printDebug("larg_mask shape:", larg_mask.shape)
            children_score_larg += larg_mask

            # rearrange children_score_rgen to index by parent
            # and argument rather than functor and argument
            # dim: imax x batch_size x cpar x p x cgen x p
            children_score_rarg = torch.gather(
                children_score_rgen, dim=2, index=lfunc_ixs
            )
            # block impossible parent-argument combinations
            # original dim of rarg_mask: cpar x cgen
            rarg_mask = self.rarg_mask.unsqueeze(1).unsqueeze(-1)
            # dim: cpar x p x cgen x p
            rarg_mask = rarg_mask.repeat(1, self.p, 1, self.p)
            # dim: qpar x qgen
            rarg_mask = rarg_mask.reshape(self.qpar, self.qgen)
            children_score_rarg += rarg_mask
            
            printDebug("scores_larg shape:", scores_larg.shape)
            printDebug("children_score_larg shape:", children_score_larg.shape)

            # dim: imax x batch_size x cpar x p x cgen x p
            y_larg = scores_larg + children_score_larg
            y_rarg = scores_rarg + children_score_rarg
            printDebug("y_larg shape:", y_larg.shape)
            printDebug("y_larg:", y_larg)
            printDebug("y_rarg:", y_rarg)

            # combine left and right arg probabilities
            # dim: imax x batch_size x cpar x p x cgen x p
            y1 = torch.logsumexp(torch.stack([y_larg, y_rarg]), dim=0)
            # dim: imax x batch_size x qpar x qgen
            printDebug("y1 shape:", y1.shape)
            y1 = y1.reshape(imax, batch_size, self.qpar, self.qgen)
            printDebug("y1 shape:", y1.shape)
            # marginalize over gen categories
            y1 = torch.logsumexp(y1, dim=3)
            # dim: imax x batch_size x cpar x p
            y1 = y1.reshape(imax, batch_size, self.cpar, self.p)
            printDebug("y1 shape:", y1.shape)

            # before this, y1 just contains probabilities for the qpar
            # possible parents.
            # But left_chart and right_chart maintain all Qall
            # categories, so pad y1 to get it up to that size

            # dim: imax x batch_size x call x p
            y1_expanded = torch.full(
                (imax, batch_size, self.call, self.p), fill_value=-QUASI_INF
            ).to(self.device)
            printDebug("y1_expanded shape:", y1_expanded.shape)

            # indices for predcats that can be parents
            # dim: cpar
            par_ixs = torch.tensor(
                [self.ix2cat.inv[c] for c in self.ix2cat_par.values()]
            ).to(self.device)
            # dim: imax x batch_size x cpar x p
            par_ixs = par_ixs.unsqueeze(-1).repeat(
                imax, batch_size, 1, self.p
            )
            # dim: imax x batch_size x call x p
            y1_expanded = y1_expanded.scatter(dim=-1, index=par_ixs, src=y1)
            # dim: imax x batch_size x call*p
            y1_expanded = y1_expanded.reshape(imax, batch_size, self.call*self.p)
            left_chart[height, imin:imax] = y1_expanded
            right_chart[height, jmin:jmax] = y1_expanded
        return left_chart, None


    def likelihood_from_chart(self, left_chart, loss_type):
        sent_len = self.this_sent_len
        # dim: batch_size x Qall
        topnode_pdf = left_chart[sent_len-1, 0]
        # dim: batch_size x Qall
        p_topnode = topnode_pdf + self.root_scores
        if loss_type == "marginal":
            logprobs = torch.logsumexp(p_topnode, dim=1)
        else:
            assert loss_type == "best_parse"
            logprobs, _ = torch.max(p_topnode, dim=1)
        return logprobs


    def compute_inside_bestparse(self, sents):
        try:
            self.this_sent_len = len(sents[0])
        except:
            print(sents)
            raise
        batch_size = len(sents)
        sent_len = self.this_sent_len

        left_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.qall)
        ).float().to(self.device)
        right_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.qall)
        ).float().to(self.device)
        # TODO make backtrack_chart a tensor
        #backtrack_chart = {}
        # backtrack_chart[ijdiff, i, s, a] is the most probable kbc
        # (split point, left child, right child) for parent category a
        # spanning words i...(i+ijdiff) in sentence s
        backtrack_chart = torch.zeros(
            sent_len, sent_len, batch_size, self.qpar, 3
        ).int().to(self.device)
        self.set_lexical_prob(sents, left_chart)
        right_chart[0] = left_chart[0]

        for ij_diff in range(1, sent_len):
            imin = 0
            imax = sent_len - ij_diff
            jmin = ij_diff
            jmax = sent_len
            height = ij_diff

            # a square of the left chart
            # dim: height x imax x batch_size x qall
            b = left_chart[0:height, imin:imax]
            # dim: height x imax x batch_size x qall
            c = torch.flip(right_chart[0:height, jmin:jmax], dims=[0])


            gen_ixs = list()
            for gen_pc_ix in range(self.qgen):
                gen_c_ix = gen_pc_ix // self.p
                p_ix = gen_pc_ix % self.p
                all_c_ix = self.ix2cat.inv[self.ix2cat_gen[gen_c_ix]]
                all_pc_ix = all_c_ix*self.p + p_ix
                gen_ixs.append(all_pc_ix)
            # dim: qgen
            gen_ixs = torch.tensor(gen_ixs).to(self.device)
            # dim: height x imax x batch_size x qgen
            gen_ixs = gen_ixs.repeat(height, imax, batch_size, 1)

            # dim: height x imax x batch_size x qgen
            #b_gen = b.gather(dim=-1, index=gen_ixs)
            b_gen = b.clone().gather(dim=-1, index=gen_ixs)
            # probability of argument i on the left followed by functor j
            # on the right
            # dim: height x imax x batch_size x qall x qgen
            children_score_lgen = b_gen[...,None,:]+c[...,None]

            # dim: height x imax x batch_size x cgen x p
            c_gen = c.gather(dim=-1, index=gen_ixs)
            # probability of functor i on the left followed by argument j
            # on the right
            # dim: height x imax x batch_size x qall x qgen
            children_score_rgen = b[...,None]+c_gen[...,None,:]

            # probability that parent category i branches into left argument j
            # and right functor i-aj
            # dim: height x imax x batch_size x qpar x qgen
            scores_larg = self.log_G_Aa.to(self.device).repeat(
                height, imax, batch_size, 1, 1
            )
            # probability that parent category i branches into left functor i-bj
            # and right argument j
            # dim: height x imax x batch_size x qpar x qgen
            scores_rarg = self.log_G_Ab.to(self.device).repeat(
                height, imax, batch_size, 1, 1
            )
            # dim: height x imax x batch_size x cpar x cgen
            rfunc_ixs = self.rfunc_ixs.repeat(height, imax, batch_size, 1, 1)

            # dim: height x imax x batch_size x cpar x cgen
            lfunc_ixs = self.lfunc_ixs.repeat(height, imax, batch_size, 1, 1)

            # rearrange children_score_lgen to index by parent
            # and argument rather than functor and argument
            # dim: height x imax x batch_size x qpar x qgen
            children_score_larg = torch.gather(
                children_score_lgen, dim=3, index=rfunc_ixs
            )
            # block impossible parent-argument combinations
            # original dim of larg_mask: cpar x cgen
            larg_mask = self.larg_mask.unsqueeze(1).unsqueeze(-1)
            # dim: cpar x p x cgen x p
            larg_mask = larg_mask.repeat(1, self.p, 1, self.p)
            # dim: qpar x qgen
            larg_mask = larg_mask.reshape(self.qpar, self.qgen)
            children_score_larg += larg_mask

            # rearrange children_score_rgen to index by parent
            # and argument rather than functor and argument
            # dim: height x imax x batch_size x cpar x p x cgen x p
            children_score_rarg = torch.gather(
                children_score_rgen, dim=3, index=lfunc_ixs
            )
            # block impossible parent-argument combinations
            # original dim of rarg_mask: cpar x cgen
            rarg_mask = self.rarg_mask.unsqueeze(1).unsqueeze(-1)
            # dim: cpar x p x cgen x p
            rarg_mask = rarg_mask.repeat(1, self.p, 1, self.p)
            # dim: qpar x qgen
            rarg_mask = rarg_mask.reshape(self.qpar, self.qgen)
            children_score_rarg += rarg_mask

            # probability that parent category i branches into left argument j
            # and right functor i-aj, that category j spans the words on the
            # left, and that category i-aj spans the words on the right
            # dim: height x imax x batch_size x qpar x qgen
            combined_scores_larg = scores_larg + children_score_larg
            # probability that parent category i branches into left functor
            # i-bj and right argument j, that category i-bj spans the words on
            # the left, and that category j spans the words on the right
            # dim: height x imax x batch_size x qpar x qgen
            combined_scores_rarg = scores_rarg + children_score_rarg
            combined_scores_larg = combined_scores_larg.permute(1,2,3,0,4)
            # dim: imax x batch_size x qpar x height*qgen
            combined_scores_larg = combined_scores_larg.contiguous().view(
                imax, batch_size, self.qpar, -1
            )
            combined_scores_rarg = combined_scores_rarg.permute(1,2,3,0,4)
            # dim: imax x batch_size x qpar x height*qgen
            combined_scores_rarg = combined_scores_rarg.contiguous().view(
                imax, batch_size, self.qpar, -1
            )
            printDebug("combined_scores_larg shape:", combined_scores_larg.shape)
            printDebug("combined_scores_rarg shape:", combined_scores_rarg.shape)

            # dim: imax x batch_size x qpar
            lmax_kbc, largmax_kbc = torch.max(combined_scores_larg, dim=3)
            rmax_kbc, rargmax_kbc = torch.max(combined_scores_rarg, dim=3)

            # dim: imax x batch_size x qpar
            l_ks = torch.div(largmax_kbc, self.qgen, rounding_mode="floor") \
                   + torch.arange(1, imax+1)[:, None, None]. to(self.device)

            # indices of argument (cat, pred) pairs
            # uses indexing from ix2cat_gen
            # dim: imax x batch_size x qpar
            l_bs = largmax_kbc % self.qgen
            printDebug("l_bs:", l_bs)

            # converts (cat, pred) pair index from qgen to qall
            qgen_2_qall = list()
            for i in range(self.qgen):
                pix = i % self.p
                cix = i // self.p
                cix_all = self.ix2cat.inv[self.ix2cat_gen[cix]]
                ix_all = cix_all*self.p + pix
                qgen_2_qall.append(ix_all)
            qgen_2_qall = torch.tensor(qgen_2_qall)

            # dim: imax x batch_size x qgen
            qgen_2_qall = qgen_2_qall.repeat(imax, batch_size, 1)
            printDebug("qgen_2_qall:", qgen_2_qall)

            # dim: imax x batch_size x qpar
            # now indexing comes from ix2cat instead of ix2cat_gen
            # This is necessary for so that l_cs and l_bs
            # use the same indexing for viterbi_backtrack
            l_bs_reindexed = torch.gather(qgen_2_qall, dim=-1, index=l_bs)
            printDebug("l_bs_reindexed:", l_bs_reindexed)

            # dim: imax x batch_size x qpar x 1
            l_bs_reshape = l_bs.view(imax, batch_size, self.qpar, 1)
            
            # dim: imax x batch_size x qpar x qgen
            rfunc_ixs = self.rfunc_ixs.repeat(imax, batch_size, 1, 1)

            # the index here tells both the category and predicate
            l_cs = torch.gather(rfunc_ixs, index=l_bs_reshape, dim=3)
            # dim: imax x batch_size x qpar
            l_cs = l_cs.squeeze(dim=3)

            # dim: 3 x imax x batch_size x qpar
            l_kbc = torch.stack([l_ks, l_bs_reindexed, l_cs], dim=0)

            # dim: imax x batch_size x qpar
            r_ks = torch.div(rargmax_kbc, self.qgen, rounding_mode="floor") \
                   + torch.arange(1, imax+1)[:, None, None]. to(self.device)
            # dim: imax x batch_size x qpar
            r_cs = rargmax_kbc % self.qgen

            # dim: imax x batch_size x qpar
            r_cs_reindexed = torch.gather(qgen_2_qall, dim=-1, index=r_cs)

            # dim: imax x batch_size x qpar x 1
            r_cs_reshape = r_cs.view(imax, batch_size, self.qpar, 1)

            # dim: imax x batch_size x qpar x qgen
            lfunc_ixs = self.lfunc_ixs.repeat(imax, batch_size, 1, 1)

            r_bs = torch.gather(lfunc_ixs, index=r_cs_reshape, dim=3)
            # dim: imax x batch_size x qpar
            r_bs = r_bs.squeeze(dim=3)

            # dim: 3 x imax x batch_size x qpar
            r_kbc = torch.stack([r_ks, r_bs, r_cs_reindexed], dim=0)

            # dim: 2 x 3 x imax x batch_size x qpar
            lr_kbc = torch.stack([l_kbc, r_kbc], dim=0)

            # dim: 2 x imax x batch_size x qpar
            lr_max = torch.stack([lmax_kbc, rmax_kbc], dim=0)

            # tells whether left arg or right arg is more likely
            # each value of the argmax is 0 (left) or 1 (right)
            # dim: imax x batch_size x qpar
            combined_max, combined_argmax = torch.max(lr_max, dim=0)

            printDebug("combined_max shape:", combined_max.shape)

            # dim: imax x batch_size x qall
            combined_max_expanded = torch.full(
                (imax, batch_size, self.qall), fill_value=-QUASI_INF
            ).to(self.device)


            # maps (cat, pred) index from qpar to qall
            qpar_2_qall = list()
            for i in range(self.qpar):
                pix = i % self.p
                cix = i // self.p
                cix_all = self.ix2cat.inv[self.ix2cat_par[cix]]
                ix_all = cix_all*self.p + pix
                qpar_2_qall.append(ix_all)
            qpar_2_qall = torch.tensor(qpar_2_qall)

            # dim: imax x batch_size x qpar
            qpar_2_qall = qpar_2_qall.repeat(imax, batch_size, 1)

            combined_max_expanded.scatter_(
                dim=-1, index=qpar_2_qall, src=combined_max
            )

            left_chart[height, imin:imax] = combined_max_expanded
            right_chart[height, jmin:jmax] = combined_max_expanded

            # gather k, b, and c
            # dim: 1 x 3 x imax x batch_size x qpar
            combined_argmax = combined_argmax.repeat(3, 1, 1, 1).unsqueeze(dim=0)
            # TODO use different arrangement of dimensions initally to
            # avoid need for permute
            # dim: 3 x imax x batch_size x qpar
            best_kbc = torch.gather(lr_kbc, index=combined_argmax, dim=0).squeeze(dim=0)
            # dim: imax x batch_size x qpar x 3
            best_kbc = best_kbc.permute(1, 2, 3, 0)
            backtrack_chart[ij_diff][:imax] = best_kbc
        return left_chart, backtrack_chart


    def viterbi_backtrack(self, left_chart, backtrack_chart, sent_index):
        sent_len = self.this_sent_len
        topnode_pdf = left_chart[sent_len-1, 0]

        # draw the top node
        p_topnode = topnode_pdf + self.root_scores
        a_ll, top_a = torch.max(p_topnode, dim=-1)
        # top_A = top_A.squeeze()
        # A_ll = A_ll.squeeze()
        printDebug("viterbi top_a likelihood:", a_ll)
        printDebug("viterbi top_a:", top_a)

        expanding_nodes = []
        expanded_nodes = []
        # rules = []
        assert self.this_sent_len > 0, "must call inside pass first!"

        a = top_a[sent_index].item()
        a_pred = a % self.p
        a_cat = a // self.p
        #a_pred, a_cat = self.ix2predcat[a]
        #A_cat_str = str(self.ix2cat[A_cat])
        a_str = "{}:{}".format(self.ix2cat[a_cat], self.ix2pred[a_pred])

        assert not ( torch.isnan(a_ll[sent_index]) \
            or torch.isinf(a_ll[sent_index]) \
                or a_ll[sent_index].item() == 0 ), \
            'something wrong with viterbi parsing. {}'.format(a_ll[sent_index])

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
            # TODO rename .cat to .predcat
            pc_ix = working_node.cat
            p_ix = pc_ix % self.p
            c_ix = pc_ix // self.p
            # convert c_ix to index from ix2cat_par instead of ix2cat
            c_par_ix = self.ix2cat_par.inv[self.ix2cat[c_ix]]
            pc_par_ix = c_par_ix*self.p + p_ix
            #pc_par_ix = self.ix2predcat_par.inv[self.ix2predcat[pc_ix]]
            k_b_c = backtrack_chart[ij_diff][ working_node.i, sent_index, pc_par_ix]
            split_point, b, c = k_b_c[0].item(), k_b_c[1].item(), k_b_c[2].item()
            #b_pred, b_cat = self.ix2predcat[b]
            b_pred = b % self.p
            b_cat = b // self.p
            #c_pred, c_cat = self.ix2predcat[c]
            c_pred = c % self.p
            c_cat = c // self.p
            b_str = "{}:{}".format(self.ix2cat[b_cat], self.ix2pred[b_pred])
            c_str = "{}:{}".format(self.ix2cat[c_cat], self.ix2pred[c_pred])

            expanded_nodes.append(working_node)
            node_b = Node(b, b_str, working_node.i, split_point, self.D, self.K, parent=working_node)
            node_c = Node(c, c_str, split_point, working_node.j, self.D, self.K, parent=working_node)
            if node_b.d == self.D and node_b.j - node_b.i != 1:
                print(node_b)
                raise Exception
            if node_b.s != 0 and node_c.s != 1:
                raise Exception("{}, {}".format(node_b, node_c))
            if node_b.is_terminal():
                expanded_nodes.append(node_b)
            else:
                expanding_nodes.append(node_b)
            if node_c.is_terminal():
                expanded_nodes.append(node_c)
            else:
                expanding_nodes.append(node_c)
        return expanded_nodes


    def set_lexical_prob(self, sent_embs, left_chart):
        lexical_probs = sent_embs.transpose(1, 0) # sentlen, batch, emb
        left_chart[0] = lexical_probs + self.opLex_probs # sentlen, batch, p
