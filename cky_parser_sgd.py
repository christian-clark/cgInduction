# import numpy as np
import torch, logging, datetime
from treenode import Node, nodes_to_tree
import torch.nn.functional as F
import numpy as np

# from .cky_parser_lcg_sparse import sparse_vit_add_trick

SMALL_NEGATIVE_NUMBER = -1e8
def logsumexp_multiply(a, b):
    max_a = a.max()
    max_b = b.max()
    res = (a - max_a).exp() @ (b - max_b).exp()
    return res.log() + max_a + max_b


# for dense grammar only! ie D must be -1
class BatchCKYParser:
    def __init__(
        self, ix2cat, lfunc_ixs, rfunc_ixs, qfunc, qres, qarg, device="cpu"
    ):
        self.D = -1
        # TODO is this correct?
        self.K = qres
        self.lexis = None # Preterminal expansion part of the grammar (this will be dense)
        self.G = None     # Nonterminal expansion part of the grammar (usually be a sparse matrix representation)
        self.p0 = None
        # self.viterbi_chart = np.zeros_like(self.chart, dtype=np.float32)
        self.ix2cat = ix2cat
        self.lfunc_ixs = lfunc_ixs
        self.rfunc_ixs = rfunc_ixs
        self.Qfunc = qfunc
        self.Qres = qres
        self.Qarg = qarg
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
            self, p0, expansion_larg, expansion_rarg, emission, pcfg_split=None
        ):
        self.log_G_larg = expansion_larg
        self.log_G_rarg = expansion_rarg
        self.log_p0 = p0
        self.log_lexis = emission
        self.pcfg_split = pcfg_split


    def marginal(self, sents, viterbi_flag=False, only_viterbi=False, sent_indices=None):
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


    def compute_inside_logspace(self, sents): #sparse
        try:
            self.this_sent_len = len(sents[0])
        except:
            print(sents)
            raise
        batch_size = len(sents)
        sent_len = self.this_sent_len

        num_points = sent_len + 1
        # left chart is the left right triangle of the chart, the top row is the lexical items, and the bottom cell is the
        #  top cell. The right chart is the left chart pushed against the right edge of the chart. The chart is a square.
        self.left_chart = torch.zeros((sent_len, sent_len, batch_size, self.Qfunc)).float().to(self.device)
        self.right_chart = torch.zeros((sent_len, sent_len, batch_size, self.Qfunc)).float().to(self.device)
        # print('lex')
        self.get_lexis_prob(sents, self.left_chart)
        self.right_chart[0] = self.left_chart[0]

        for ij_diff in range(1, sent_len):
            imin = 0
            imax = sent_len - ij_diff
            jmin = ij_diff
            jmax = sent_len
            height = ij_diff

            # a square of the left chart
            # dim: height x imax x batch_size x Qfunc
            b = self.left_chart[0:height, imin:imax]
            # dim: height x imax x batch_size x Qfunc
            c = torch.flip(self.right_chart[0:height, jmin:jmax], dims=[0])
            # TODO this can be optimized by taking advantage of the fact that
            # maximally deep categories can only appear on preterminal nodes
            # NOTE: doing the logsumexp here means this takes more memory
            # than the parallel line in compute_viterbi_inside
            # dim: imax x batch_size x Qfunc x Qfunc
            #dot_temp_mat = torch.logsumexp(b[...,None]+c[...,None,:], dim=0)
            # dot_temp_mat[..., i, j] is score for left child i, right child j


            # dim: height x imax x batch_size x Qfunc x Qarg
            children_score_larg = torch.logsumexp(
                b[...,None,:self.Qarg] + c[...,None], dim=0
            )

            # dim: height x imax x batch_size x Qfunc x Qarg
            children_score_rarg = torch.logsumexp(
                b[...,None] + c[...,None,:self.Qarg], dim=0
            )

            # scores_larg[..., i, j] is score for parent i, larg j
            # dim: imax x batch_size x Qres x Qarg
            scores_larg = self.log_G_larg.to(self.device).repeat(
                imax, batch_size, 1, 1
            )

            #print("CEC scores_larg shape: {}".format(scores_larg.shape))
            scores_rarg = self.log_G_rarg.to(self.device).repeat(
                imax, batch_size, 1, 1
            )

            # rfunc_ixs[..., i, j] tells the index of the right-child functor
            # going with parent cat i, larg cat j
            # dim: imax x batch_size x Qres x Qarg
            rfunc_ixs = self.rfunc_ixs.repeat(imax, batch_size, 1, 1)
            lfunc_ixs = self.lfunc_ixs.repeat(imax, batch_size, 1, 1)

            # dtm_permute [..., i, j] is score for right child i, left child j
            # needed to play with gather() correctly for larg case
            # dim: imax x batch_size x Qfunc x Qfunc
            #dtm_permute = dot_temp_mat.permute(0, 1, 3, 2)

            # dot_temp_mat_larg[..., i, j] is score for parent i, larg j
            # dim: imax x batch_size x Qres x Qarg
            #dot_temp_mat_larg = torch.gather(
            #    dtm_permute[..., :self.Qarg], dim=2, index=rfunc_ixs
            #)
            children_score_larg = torch.gather(
                children_score_larg, dim=2, index=rfunc_ixs
            )
            #print("CEC dot_temp_mat_larg shape: {}".format(dot_temp_mat_larg.shape))

            # dot_temp_mat_larg[..., i, j] is score for parent i, rarg j
            # dim: imax x batch_size x Qres x Qarg
            #dot_temp_mat_rarg = torch.gather(
            #    dot_temp_mat[..., :self.Qarg], dim=2, index=lfunc_ixs
            #)
            children_score_rarg = torch.gather(
                children_score_rarg, dim=2, index=lfunc_ixs
            )

            # TODO can this be done more space-efficiently?
            # dim: imax x batch_size x Qres x Qarg
            #y_larg = scores_larg + dot_temp_mat_larg
            #y_rarg = scores_rarg + dot_temp_mat_rarg
            y_larg = scores_larg + children_score_larg
            y_rarg = scores_rarg + children_score_rarg
            # combine left and right arg probabilities
            # dim: imax x batch_size x Qres x Qarg
            y1 = torch.logsumexp(torch.stack([y_larg, y_rarg]), dim=0)
            # marginalize over arg categories
            # dim: imax x batch_size x Qres
            #y1 = torch.logsumexp(y1, dim=2)
            y1 = torch.logsumexp(y1, dim=3)

            #print("CEC y1 shape: {}".format(y1.shape))

            # before this, y1 just contains probabilities for the Qres
            # result categories.
            # But left_chart and right_chart maintain all Qfunc
            # categories, so pad y1 to get it up to that size

            # TODO an alternative to padding here might be to initialize
            # left_chart and right_chart to all -inf
            QUASI_INF = 10000000.
            padding = torch.full(
                (imax, batch_size, self.Qfunc-self.Qres),
                fill_value=-QUASI_INF
            )
            padding = padding.to(self.device)
            #print("CEC padding shape: {}".format(padding.shape))

            # dim: imax x batch_size x Qfunc
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
        self.this_sent_len = sent.shape[1]
        batch_size = 1
        sent_len = self.this_sent_len
        num_points = sent_len + 1
        self.left_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.Qfunc)
        ).float().to(self.device)
        self.right_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.Qfunc)
        ).float().to(self.device)

        backtrack_chart = {}

        # print('lex')
        self.get_lexis_prob(sent, self.left_chart)
        self.right_chart[0] = self.left_chart[0]


        for ij_diff in range(1, sent_len):
            imin = 0
            imax = sent_len - ij_diff
            jmin = ij_diff
            jmax = sent_len
            height = ij_diff

            # a square of the left chart
            # dim: height x imax x batch_size x Qfunc
            b = self.left_chart[0:height, imin:imax]
            # dim: height x imax x batch_size x Qfunc
            c = torch.flip(self.right_chart[0:height, jmin:jmax], dims=[0])

            #print("CEC b")
            #print(b)
            #print("CEC c")
            #print(c)

            # dim: height x imax x batch_size x Qfunc x Qfunc
            #dot_temp_mat = ( b[...,None]+c[...,None,:] ).view(
                #height, imax, batch_size, self.Qfunc, self.Qfunc
            #)

            # dim: height x imax x batch_size x Qfunc x Qarg
            children_score_larg = b[...,None,:self.Qarg]+c[...,None]

            # dim: height x imax x batch_size x Qfunc x Qarg
            children_score_rarg = b[...,None]+c[...,None,:self.Qarg]
            #print("CEC dot_temp_mat")
            #print(dot_temp_mat)

            # scores_larg[..., i, j] is score for parent i, larg j
            # dim: height x imax x batch_size x Qres x Qarg
            scores_larg = self.log_G_larg.to(self.device).repeat(
                height, imax, batch_size, 1, 1
            )
            scores_rarg = self.log_G_rarg.to(self.device).repeat(
                height, imax, batch_size, 1, 1
            )

            #print("CEC scores_larg")
            #print(scores_larg)
            #print("CEC scores_rarg")
            #print(scores_rarg)

            # dim: height x imax x batch_size x Qres x Qarg
            rfunc_ixs = self.rfunc_ixs.repeat(height, imax, batch_size, 1, 1)
            lfunc_ixs = self.lfunc_ixs.repeat(height, imax, batch_size, 1, 1)

            #print("CEC rfunc_ixs")
            #print(rfunc_ixs)
            #print("CEC lfunc_ixs")
            #print(lfunc_ixs)

            #dtm_permute = dot_temp_mat.permute(0, 1, 2, 4, 3)

            #print("CEC dtm_permute")
            #print(dtm_permute)

            # dim: height x imax x batch_size x Qres x Qarg
            #dot_temp_mat_larg = torch.gather(
            #    dtm_permute[..., :self.Qarg], dim=3, index=rfunc_ixs
            #)

            children_score_larg = torch.gather(
                children_score_larg, dim=3, index=rfunc_ixs
            )

            #dot_temp_mat_rarg = torch.gather(
            #    dot_temp_mat[..., :self.Qarg], dim=3, index=lfunc_ixs
            #)

            children_score_rarg = torch.gather(
                children_score_rarg, dim=3, index=lfunc_ixs
            )

            #print("CEC dot_temp_mat_larg")
            #print(dot_temp_mat_larg)
            #print("CEC dot_temp_mat_rarg")
            #print(dot_temp_mat_rarg)

            # dim: height x imax x batch_size x Qres x Qarg
            #combined_scores_larg = scores_larg + dot_temp_mat_larg
            #combined_scores_rarg = scores_rarg + dot_temp_mat_rarg
            combined_scores_larg = scores_larg + children_score_larg
            combined_scores_rarg = scores_rarg + children_score_rarg

            #print("CEC combined_scores_larg")
            #print(combined_scores_larg)
            #print("CEC combined_scores_rarg")
            #print(combined_scores_rarg)

            combined_scores_larg = combined_scores_larg.permute(1,2,3,0,4)
            # dim: imax x batch_size x Qres x height*Qarg
            combined_scores_larg = combined_scores_larg.contiguous().view(
                imax, batch_size, self.Qres, -1
            )
            combined_scores_rarg = combined_scores_rarg.permute(1,2,3,0,4)
            # dim: imax x batch_size x Qres x height*Qarg
            combined_scores_rarg = combined_scores_rarg.contiguous().view(
                imax, batch_size, self.Qres, -1
            )

            #print("CEC combined_scores_larg permuted")
            #print(combined_scores_larg)
            #print("CEC combined_scores_rarg permuted")
            #print(combined_scores_rarg)


            # dim: imax x batch_size x Qres
            lmax_kbc, largmax_kbc = torch.max(combined_scores_larg, dim=3)
            rmax_kbc, rargmax_kbc = torch.max(combined_scores_rarg, dim=3)

            #print("CEC lmax_kbc")
            #print(lmax_kbc)
            #print("CEC largmax_kbc")
            #print(largmax_kbc)

            # dim: imax x batch_size x Qres
            l_ks = torch.div(largmax_kbc, self.Qarg, rounding_mode="floor") \
                   + torch.arange(1, imax+1)[:, None, None]. to(self.device)
            # dim: imax x batch_size x Qres
            l_bs = largmax_kbc % self.Qarg

            #print("CEC l_ks")
            #print(l_ks)
            #print("CEC l_bs")
            #print(l_bs)

            # dim: imax x batch_size x Qres x 1
            l_bs_reshape = l_bs.view(imax, batch_size, self.Qres, 1)

            #print("CEC l_bs_reshape:")
            #print(l_bs_reshape)

            # dim: imax x batch_size x Qres x Qarg
            rfunc_ixs = self.rfunc_ixs.repeat(
                imax, batch_size, 1, 1
            )

            #print("CEC rfunc_ixs:")
            #print(rfunc_ixs)

            # dim: imax x batch_size x Qres
            l_cs = torch.gather(rfunc_ixs, index=l_bs_reshape, dim=3).squeeze(dim=3)
 
            # dim: 3 x imax x batch_size x Qarg
            l_kbc = torch.stack([l_ks, l_bs, l_cs], dim=0)

            # dim: imax x batch_size x Qres
            r_ks = torch.div(rargmax_kbc, self.Qarg, rounding_mode="floor") \
                   + torch.arange(1, imax+1)[:, None, None]. to(self.device)
            # dim: imax x batch_size x Qres
            r_cs = rargmax_kbc % self.Qarg

            # dim: imax x batch_size x Qres x 1
            r_cs_reshape = r_cs.view(imax, batch_size, self.Qres, 1)

            # dim: imax x batch_size x Qres x Qarg
            lfunc_ixs = self.lfunc_ixs.repeat(
                imax, batch_size, 1, 1
            )

            # dim: imax x batch_size x Qres
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
                (imax, batch_size, self.Qfunc-self.Qres),
                fill_value=-QUASI_INF
            )
            padding = padding.to(self.device)

            # dim: imax x batch_size x Qfunc
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

        A_cat = top_A[sent_index].item()
        A_cat_str = str(self.ix2cat[A_cat])
        
        assert not ( torch.isnan(A_ll[sent_index]) or torch.isinf(A_ll[sent_index]) or A_ll[sent_index].item() == 0 ), \
            'something wrong with viterbi parsing. {}'.format(A_ll[sent_index])

        # prepare the downward sampling pass
        top_node = Node(A_cat, A_cat_str, 0, sent_len, self.D, self.K)
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
            b_str = str(self.ix2cat[b])
            c_str = str(self.ix2cat[c])

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
            lexis_probs = lexis_probs + self.pcfg_split[:, 1] # sentlen, batch, p
            full_lexis_probs = lexis_probs
        else:
            full_lexis_probs = torch.full((lexis_probs.shape[0], lexis_probs.shape[1], self.K), SMALL_NEGATIVE_NUMBER,
                                          device=lexis_probs.device)
            full_lexis_probs = torch.cat([full_lexis_probs, lexis_probs], dim=2)
            print(full_lexis_probs[0,0])
        left_chart[0] = full_lexis_probs
        return

