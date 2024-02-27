# import numpy as np
import torch, logging, datetime
from treenode import Node, nodes_to_tree
import torch.nn.functional as F
import numpy as np

QUASI_INF = 10000000.

DEBUG = False
def printDebug(*args, **kwargs):
    if DEBUG:
        print("DEBUG: ", end="")
        print(*args, **kwargs)


def logsumexp_multiply(a, b):
    max_a = a.max()
    max_b = b.max()
    res = (a - max_a).exp() @ (b - max_b).exp()
    return res.log() + max_a + max_b


# for dense grammar only! ie D must be -1
class BatchCKYParser:
    def __init__(
        self, ix2cat, ix2pred, ix2predcat_all, ix2predcat_par, ix2predcat_gen, 
        genpc_2_pc, limp_ixs, rimp_ixs, qall, qpar,
        qgen, device="cpu"
    ):
        # TODO figure out/document what D and K do
        self.D = -1
        self.K = qpar
        self.ix2cat = ix2cat
        self.ix2pred = ix2pred
        self.ix2predcat_all = ix2predcat_all
        self.ix2predcat_par = ix2predcat_par
        self.ix2predcat_gen = ix2predcat_gen
        self.genpc_2_pc = genpc_2_pc
        self.limp_ixs = limp_ixs
        self.rimp_ixs = rimp_ixs
        # total number of predcats, i.e. (predicate, category) pairs
        self.Qall = qall
        # total number of parent predcats
        self.Qpar = qpar
        # total number of argument predcats
        self.Qgen = qgen
        self.this_sent_len = -1
        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'


    def set_models(
            self, p0, expansion_lgen, expansion_rgen, split_scores
        ):
        self.log_G_lgen = expansion_lgen
        self.log_G_rgen = expansion_rgen
        self.log_p0 = p0
        self.split_scores = split_scores

    
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
            (sent_len, sent_len, batch_size, self.Qall)
        ).float().to(self.device)
        right_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.Qall)
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
            # dim: height x imax x batch_size x Qall
            b = left_chart[0:height, imin:imax]
            # dim: height x imax x batch_size x Qall
            c = torch.flip(right_chart[0:height, jmin:jmax], dims=[0])
            
            # indices for predcats that can be arguments
            gen_ixs = torch.tensor(
                [self.ix2predcat_all.inv[pc] for pc in self.ix2predcat_gen.values()]
            ).to(self.device)
            # dim: height x imax x batch_size x Qgen
            gen_ixs = gen_ixs.repeat(
                height, imax, batch_size, 1
            )

            # TODO this can be optimized by taking advantage of the fact that
            # maximally deep categories can only appear on preterminal nodes
            # NOTE: doing the logsumexp here means this takes more memory
            # than the parallel line in compute_viterbi_inside

            # dim: height x imax x batch_size x Qgen
            # torch throws an error about inplace modification if the clone()
            # isn't there...idk why
            b_gen = b.clone().gather(dim=-1, index=gen_ixs)
            # probability of argument i on the left followed by functor j
            # on the right
            # dim: height x imax x batch_size x Qall x Qgen
            children_score_lgen = torch.logsumexp(
                b_gen[...,None,:] + c[...,None], dim=0
            )

            # dim: height x imax x batch_size x Qgen
            c_gen = c.gather(dim=-1, index=gen_ixs)
            # probability of functor i on the left followed by argument j
            # on the right
            # dim: height x imax x batch_size x Qall x Qgen
            children_score_rgen = torch.logsumexp(
                b[...,None] + c_gen[...,None,:], dim=0
            )

            # probability that parent category i branches into left argument j
            # and right functor i-aj
            # dim: imax x batch_size x Qpar x Qgen
            scores_larg = self.log_G_lgen.to(self.device).repeat(
                imax, batch_size, 1, 1
            )
            # probability that parent category i branches into left functor i-bj
            # and right argument j
            # dim:  imax x batch_size x Qpar x Qgen
            scores_rarg = self.log_G_rgen.to(self.device).repeat(
                imax, batch_size, 1, 1
            )

            # dim: imax x batch_size x Qpar x Qgen
            rimp_ixs = self.rimp_ixs.repeat(imax, batch_size, 1, 1)

            # dim: imax x batch_size x Qpar x Qgen
            limp_ixs = self.limp_ixs.repeat(imax, batch_size, 1, 1)

            # rearrange children_score_lgen to index by parent
            # and argument rather than functor and argument
            # dim: height x imax x batch_size x Qpar x Qgen
            children_score_larg = torch.gather(
                children_score_lgen, dim=2, index=rimp_ixs
            )
            # block impossible parent-argument combinations
            # NOTE this was moved to cg_inducer.forward
            #children_score_larg += self.lgen_mask

            # rearrange children_score_rgen to index by parent
            # and argument rather than functor and argument
            # dim: height x imax x batch_size x Qpar x Qgen
            children_score_rarg = torch.gather(
                children_score_rgen, dim=2, index=limp_ixs
            )
            # block impossible parent-argument combinations
            # NOTE this was moved to cg_inducer.forward
            #children_score_rarg += self.rgen_mask

            # TODO can this be done more space-efficiently?
            # dim: imax x batch_size x Qpar x Qgen
            y_larg = scores_larg + children_score_larg
            y_rarg = scores_rarg + children_score_rarg

            # combine left and right arg probabilities
            # dim: imax x batch_size x Qpar x Qgen
            y1 = torch.logsumexp(torch.stack([y_larg, y_rarg]), dim=0)
            # marginalize over gen categories
            # dim: imax x batch_size x Qpar
            y1 = torch.logsumexp(y1, dim=3)

            # before this, y1 just contains probabilities for the Qpar
            # parent categories.
            # But left_chart and right_chart maintain all Qall
            # categories, so pad y1 to get it up to that size

            # dim: imax x batch_size x Qall
            y1_expanded = torch.full(
                (imax, batch_size, self.Qall), fill_value=-QUASI_INF
            ).to(self.device)

            # indices for predcats that can be parents
            par_ixs = torch.tensor(
                [self.ix2predcat_all.inv[pc] for pc in self.ix2predcat_par.values()]
            ).to(self.device)
            # dim: imax x batch_size x Qpar
            par_ixs = par_ixs.repeat(
                imax, batch_size, 1
            )
            # dim: imax x batch_size x Qall
            y1_expanded = y1_expanded.scatter(dim=-1, index=par_ixs, src=y1)
            left_chart[height, imin:imax] = y1_expanded
            right_chart[height, jmin:jmax] = y1_expanded
        return left_chart, None


    def likelihood_from_chart(self, left_chart, loss_type):
        sent_len = self.this_sent_len
        # dim: batch_size x Qall
        topnode_pdf = left_chart[sent_len-1, 0]
        # dim: batch_size x Qall
        p_topnode = topnode_pdf + self.log_p0
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
            (sent_len, sent_len, batch_size, self.Qall)
        ).float().to(self.device)
        right_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.Qall)
        ).float().to(self.device)
        # TODO make backtrack_chart a tensor
        #backtrack_chart = {}
        # backtrack_chart[ijdiff, i, s, a] is the most probable kbc
        # (split point, left child, right child) for parent category a
        # spanning words i...(i+ijdiff) in sentence s
        backtrack_chart = torch.zeros(
            sent_len, sent_len, batch_size, self.Qpar, 3
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
            # dim: height x imax x batch_size x Qall
            b = left_chart[0:height, imin:imax]
            # dim: height x imax x batch_size x Qall
            c = torch.flip(right_chart[0:height, jmin:jmax], dims=[0])

            # indices for predcats that can be arguments
            gen_ixs = torch.tensor(
                [self.ix2predcat_all.inv[pc] for pc in self.ix2predcat_gen.values()]
            ).to(self.device)
            # dim: height x imax x batch_size x Qgen
            gen_ixs = gen_ixs.repeat(
                height, imax, batch_size, 1
            )

            # dim: height x imax x batch_size x Qgen
            #b_gen = b.gather(dim=-1, index=gen_ixs)
            b_gen = b.clone().gather(dim=-1, index=gen_ixs)
            # probability of argument i on the left followed by functor j
            # on the right
            # dim: height x imax x batch_size x Qall x Qgen
            children_score_lgen = b_gen[...,None,:]+c[...,None]

            # dim: height x imax x batch_size x Qgen
            c_gen = c.gather(dim=-1, index=gen_ixs)
            # probability of functor i on the left followed by argument j
            # on the right
            # dim: height x imax x batch_size x Qall x Qgen
            children_score_rgen = b[...,None]+c_gen[...,None,:]

            # probability that parent category i branches into left argument j
            # and right functor i-aj
            # dim: height x imax x batch_size x Qpar x Qgen
            scores_larg = self.log_G_lgen.to(self.device).repeat(
                height, imax, batch_size, 1, 1
            )
            # probability that parent category i branches into left functor i-bj
            # and right argument j
            # dim: height x imax x batch_size x Qpar x Qgen
            scores_rarg = self.log_G_rgen.to(self.device).repeat(
                height, imax, batch_size, 1, 1
            )

            # dim: height x imax x batch_size x Qpar x Qgen
            rimp_ixs = self.rimp_ixs.repeat(height, imax, batch_size, 1, 1)

            # dim: height x imax x batch_size x Qpar x Qgen
            limp_ixs = self.limp_ixs.repeat(height, imax, batch_size, 1, 1)

            # rearrange children_score_lgen to index by parent
            # and argument rather than functor and argument
            # dim: height x imax x batch_size x Qpar x Qgen
            children_score_larg = torch.gather(
                children_score_lgen, dim=3, index=rimp_ixs
            )
            # block impossible parent-argument combinations
            #children_score_larg += self.lgen_mask

            # rearrange children_score_rgen to index by parent
            # and argument rather than functor and argument
            # dim: height x imax x batch_size x Qpar x Qgen
            children_score_rarg = torch.gather(
                children_score_rgen, dim=3, index=limp_ixs
            )
            # block impossible parent-argument combinations
            #children_score_rarg += self.rgen_mask

            # probability that parent category i branches into left argument j
            # and right functor i-aj, that category j spans the words on the
            # left, and that category i-aj spans the words on the right
            # dim: height x imax x batch_size x Qpar x Qarg
            combined_scores_larg = scores_larg + children_score_larg
            # probability that parent category i branches into left functor
            # i-bj and right argument j, that category i-bj spans the words on
            # the left, and that category j spans the words on the right
            # dim: height x imax x batch_size x Qpar x Qgen
            combined_scores_rarg = scores_rarg + children_score_rarg

            combined_scores_larg = combined_scores_larg.permute(1,2,3,0,4)
            # dim: imax x batch_size x Qpar x height*Qgen
            combined_scores_larg = combined_scores_larg.contiguous().view(
                imax, batch_size, self.Qpar, -1
            )
            combined_scores_rarg = combined_scores_rarg.permute(1,2,3,0,4)
            # dim: imax x batch_size x Qpar x height*Qgen
            combined_scores_rarg = combined_scores_rarg.contiguous().view(
                imax, batch_size, self.Qpar, -1
            )

            # dim: imax x batch_size x Qpar
            lmax_kbc, largmax_kbc = torch.max(combined_scores_larg, dim=3)
            rmax_kbc, rargmax_kbc = torch.max(combined_scores_rarg, dim=3)

            # dim: imax x batch_size x Qpar
            l_ks = torch.div(largmax_kbc, self.Qgen, rounding_mode="floor") \
                   + torch.arange(1, imax+1)[:, None, None]. to(self.device)

            # NOTE: these are the predcat indices based on the indexing for
            # argument predcats
            # dim: imax x batch_size x Qpar
            l_bs = largmax_kbc % (self.Qgen)

            # dim: imax x batch_size x Qpar x 1
            l_bs_reshape = l_bs.view(imax, batch_size, self.Qpar, 1)

            # dim: imax x batch_size x Qpar
            rimp_ixs = rimp_ixs[0]
            l_cs = torch.gather(rimp_ixs, index=l_bs_reshape, dim=3).squeeze(dim=3)

            # dim: imax x batch_size x Qgen
            pc_ix = self.genpc_2_pc.repeat(imax, batch_size, 1)

            # dim: imax x batch_size x Qpar
            # now each entry is an index for ix2predcat_all instead of
            # ix2predcat_gen. This is necessary for so that l_cs and l_bs
            # use the same indexing for viterbi_backtrack
            l_bs_reindexed = torch.gather(pc_ix, dim=-1, index=l_bs)

            # dim: 3 x imax x batch_size x par
            l_kbc = torch.stack([l_ks, l_bs_reindexed, l_cs], dim=0)

            # dim: imax x batch_size x Qpar
            r_ks = torch.div(rargmax_kbc, self.Qgen, rounding_mode="floor") \
                   + torch.arange(1, imax+1)[:, None, None]. to(self.device)
            # dim: imax x batch_size x Qpar
            r_cs = rargmax_kbc % (self.Qgen)

            # dim: imax x batch_size x Qpar x 1
            r_cs_reshape = r_cs.view(imax, batch_size, self.Qpar, 1)

            # dim: imax x batch_size x Qres
            limp_ixs = limp_ixs[0]
            r_bs = torch.gather(limp_ixs, index=r_cs_reshape, dim=3).squeeze(dim=3)
            # dim: imax x batch_size x Qpar
            r_cs_reindexed = torch.gather(pc_ix, dim=-1, index=r_cs)

            # dim: 3 x imax x batch_size x Qpar
            r_kbc = torch.stack([r_ks, r_bs, r_cs_reindexed], dim=0)

            # dim: 2 x 3 x imax x batch_size x Qpar
            lr_kbc = torch.stack([l_kbc, r_kbc], dim=0)

            # dim: 2 x imax x batch_size x Qpar
            lr_max = torch.stack([lmax_kbc, rmax_kbc], dim=0)

            # tells whether left arg or right arg is more likely
            # each value of the argmax is 0 (left) or 1 (right)
            # dim: imax x batch_size x Qpar
            combined_max, combined_argmax = torch.max(lr_max, dim=0)

            # dim: imax x batch_size x Qall
            combined_max_expanded = torch.full(
                (imax, batch_size, self.Qall), fill_value=-QUASI_INF
            ).to(self.device)
            # indices for predcats that can be parents
            par_ixs = torch.tensor(
                [self.ix2predcat_all.inv[pc] for pc in self.ix2predcat_par.values()]
            ).to(self.device)
            # dim: imax x batch_size x Qpar
            par_ixs = par_ixs.repeat(
                imax, batch_size, 1
            )
            combined_max_expanded.scatter_(dim=-1, index=par_ixs, src=combined_max)

            left_chart[height, imin:imax] = combined_max_expanded
            right_chart[height, jmin:jmax] = combined_max_expanded

            # gather k, b, and c
            # dim: 1 x 3 x imax x batch_size x Qpar
            combined_argmax = combined_argmax.repeat(3, 1, 1, 1).unsqueeze(dim=0)
            # TODO use different arrangement of dimensions initally to
            # avoid need for permute
            # dim: 3 x imax x batch_size x Qpar
            best_kbc = torch.gather(lr_kbc, index=combined_argmax, dim=0).squeeze(dim=0)
            # dim: imax x batch_size x Qpar x 3
            best_kbc = best_kbc.permute(1, 2, 3, 0)
            backtrack_chart[ij_diff][:imax] = best_kbc
        return left_chart, backtrack_chart


    def viterbi_backtrack(self, left_chart, backtrack_chart, sent_index):
        sent_len = self.this_sent_len
        topnode_pdf = left_chart[sent_len-1, 0]

        # draw the top node
        p_topnode = topnode_pdf + self.log_p0
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
        a_pred, a_cat = self.ix2predcat_all[a]
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
            pc_par_ix = self.ix2predcat_par.inv[self.ix2predcat_all[pc_ix]]
            k_b_c = backtrack_chart[ij_diff][ working_node.i, sent_index, pc_par_ix]
            split_point, b, c = k_b_c[0].item(), k_b_c[1].item(), k_b_c[2].item()
            b_pred, b_cat = self.ix2predcat_all[b]
            c_pred, c_cat = self.ix2predcat_all[c]
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
        left_chart[0] = lexical_probs + self.split_scores[:, 1] # sentlen, batch, p
