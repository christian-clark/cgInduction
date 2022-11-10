# import numpy as np
import torch, logging, datetime
from treenode import Node, nodes_to_tree
import torch.nn.functional as F

# from .cky_parser_lcg_sparse import sparse_vit_add_trick

SMALL_NEGATIVE_NUMBER = -1e8
def logsumexp_multiply(a, b):
    max_a = a.max()
    max_b = b.max()
    res = (a - max_a).exp() @ (b - max_b).exp()
    return res.log() + max_a + max_b


# for dense grammar only! ie D must be -1
class batch_CKY_parser:
    def __init__(
        self, ix2cat, l2r, r2l,
        nt=0, t=0, device='cpu'
    ):
        self.D = -1
        self.K = nt
        self.lexis = None # Preterminal expansion part of the grammar (this will be dense)
        self.G = None     # Nonterminal expansion part of the grammar (usually be a sparse matrix representation)
        self.p0 = None
        # self.viterbi_chart = np.zeros_like(self.chart, dtype=np.float32)
        self.ix2cat = ix2cat
        self.l2r = l2r
        self.r2l = r2l
        self.Q = nt + t
        self.this_sent_len = -1
        self.counter = 0
        self.vocab_prob_list = []
        self.finished_vocab = set()
        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def set_models(self, p0, expansion_l, expansion_r, emission, pcfg_split=None):
        #self.log_G = expansion
        self.log_G_l = expansion_l
        self.log_G_r = expansion_r
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

    # @profile
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
        self.left_chart = torch.zeros((sent_len, sent_len, batch_size, self.Q)).float().to(self.device)
        self.right_chart = torch.zeros((sent_len, sent_len, batch_size, self.Q)).float().to(self.device)
        # print('lex')
        self.get_lexis_prob(sents, self.left_chart)
        self.right_chart[0] = self.left_chart[0]

        for ij_diff in range(1, sent_len):
            left_min = 0
            left_max = sent_len - ij_diff
            right_min = ij_diff
            right_max = sent_len
            height = ij_diff

            b = self.left_chart[0:height, left_min:left_max] # (all_ijdiffs, i-j, batch, Q) a square of the left chart
            c = torch.flip(self.right_chart[0:height, right_min:right_max], dims=[0])
            #
            dot_temp_mat = torch.logsumexp(b[...,None]+c[...,None,:], dim=0).view(left_max, batch_size, self.Q, self.Q)

            # add an extra state to the last two dimensions to deal with NULL
            QUASI_INF = 10000000
            null_tensor = torch.tensor([-QUASI_INF])
            null_col_1 = null_tensor.reshape(1, 1, 1, 1).expand(left_max, batch_size, self.Q, 1)
            dot_temp_mat = torch.cat([dot_temp_mat, null_col_1], dim=3)
            # dot_temp_mat is now left_max x batch_size x Q x (Q+1)
            null_col_2 = null_tensor.reshape(1, 1, 1, 1).expand(left_max, batch_size, 1, self.Q+1)
            dot_temp_mat = torch.cat([dot_temp_mat, null_col_2], dim=2)
            # dot_temp_mat is now left_max x batch_size x (Q+1) x (Q+1)


            scores_l = self.log_G_l.to(self.device).repeat(left_max, batch_size, 1, 1)
            # left_max x batch_size x Q x Q

            #scores_r = F.log_softmax(self.log_G_r*self.rule_filter_r, dim=1) # Q x Q
            scores_r = self.log_G_r.to(self.device).repeat(left_max, batch_size, 1, 1)

            # left_max x batch_size x Q x Q

            l2r_stacked = self.l2r.unsqueeze(dim=1).repeat(
                left_max, batch_size, 1, 1
            ).to(self.device)
            # left_max x batch_size x Q

            r2l_stacked = self.r2l.unsqueeze(dim=0).repeat(
                left_max, batch_size, 1, 1
            ).to(self.device)

            # left_max x batch_size x Q
            # throw out the last column for NULL
            l_functor_prob = torch.gather(dot_temp_mat, dim=3, index=l2r_stacked).squeeze(dim=3)[..., :self.Q]
            # left_max x batch_size x Q
            r_functor_prob = torch.gather(dot_temp_mat, dim=2, index=r2l_stacked).squeeze(dim=2)[..., :self.Q]

            # TODO can this be done without creating all the QxQ matrices?
            y_l = scores_l + l_functor_prob[...,None,:]
            y_r = scores_r + r_functor_prob[...,None,:]
            y1 = torch.logsumexp(torch.stack([y_l, y_r]), dim=0)
            #y1 = y_l
            # left_max x batch_size x Q x Q
            y1 = torch.logsumexp(y1, dim=3)

            self.left_chart[height, left_min:left_max] = y1
            self.right_chart[height, right_min:right_max] = y1
        return

    def compute_inside_logspace_pykeops(self, sents): #sparse
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
        self.left_chart = torch.zeros((sent_len, sent_len, batch_size, self.Q)).float().to(self.device)
        self.right_chart = torch.zeros((sent_len, sent_len, batch_size, self.Q)).float().to(self.device)
        # print('lex')
        if self.device == 'cpu':
            pykeops_device = 'CPU'
        else:
            pykeops_device = 'GPU'

        self.get_lexis_prob(sents, self.left_chart)
        self.right_chart[0] = self.left_chart[0]

        for ij_diff in range(1, sent_len):

            left_min = 0
            left_max = sent_len - ij_diff
            right_min = ij_diff
            right_max = sent_len
            height = ij_diff

            b = self.left_chart[0:height, left_min:left_max].permute(1, 2, 3, 0).contiguous() # (all_ijdiffs, i-j, batch, Q) a square of the left chart
            c = torch.flip(self.right_chart[0:height, right_min:right_max], dims=[0]).permute(1, 2, 3, 0).contiguous()
            #
            if height > 1:
                b = LazyTensor(b.view(left_max - left_min, batch_size, self.Q, 1, 1, height, 1))  # (i-j, batch, Q, Q, height, 1)
                c = LazyTensor(c.view(left_max - left_min, batch_size, 1, self.Q, 1, height, 1))
                # print(b.shape)
                # print((b+c).shape)
                dot_temp_mat = (b + c).logsumexp(dim=5, backend=pykeops_device)  # (i-j, batch, Q, Q, 1)
                dot_temp_mat = dot_temp_mat.view(sent_len - ij_diff, batch_size, 1, 1, -1, 1)  # (i-j, batch, 1, Q**2, 1)
                # print(dot_temp_mat.shape)
            else:
                dot_temp_mat = torch.logsumexp(b[..., None, :] + c[..., None, :, :], dim=4).view(sent_len - ij_diff, batch_size, -1)

            # dot_temp_mat = torch.logsumexp(b[..., None] + c[..., None, :], dim=0)
            # i-j, batch, Q**2
            # dense
            dot_temp_mat = LazyTensor(dot_temp_mat.view(sent_len - ij_diff, batch_size, 1, 1, -1, 1))  # (i-j, batch, 1, Q**2, 1)
            # print(dot_temp_mat.shape)
            log_G_lazy = LazyTensor(self.log_G.view(1, 1, self.Q, 1, self.Q ** 2, 1))  # (1, 1, Q, Q**2, 1)
            filtered_lazy = log_G_lazy + dot_temp_mat  # (i-j, batch, Q, Q**2, 1)
            # print(filtered_lazy.shape)
            y1 = filtered_lazy.logsumexp(4, backend=pykeops_device).squeeze(-1).squeeze(-1)  # i-j, batch, Q

            self.left_chart[height, left_min:left_max] = y1
            self.right_chart[height, right_min:right_max] = y1
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

    # @profile
    def compute_viterbi_inside(self, sent): #sparse
        self.this_sent_len = sent.shape[1]
        batch_size = 1
        # sents_tensor = torch.tensor(sents).to('cuda')
        sent_len = self.this_sent_len
        num_points = sent_len + 1
        self.left_chart = torch.zeros((sent_len, sent_len, batch_size, self.Q)).float().to(self.device)
        self.right_chart = torch.zeros((sent_len, sent_len, batch_size, self.Q)).float().to(self.device)

        # viterbi_chart = np.full((num_points, num_points, batch_size, self.Q), -np.inf, dtype=np.float)
        backtrack_chart = {}

        # print('lex')
        self.get_lexis_prob(sent, self.left_chart)
        # self._find_max_prob_words_within_vit(sent, self.left_chart[0])
        self.right_chart[0] = self.left_chart[0]

        # kron_temp_mat = np.zeros((batch_size, self.Q ** 2), dtype=np.float)
        # kron_temp_mat_3d_view = kron_temp_mat.reshape((batch_size, self.Q, self.Q))

        for ij_diff in range(1, sent_len):
            # print('ijdiff', ij_diff)
            left_min = 0
            left_max = sent_len - ij_diff
            right_min = ij_diff
            right_max = sent_len
            height = ij_diff

            b = self.left_chart[0:height, left_min:left_max] # (all_ijdiffs, i-j, batch, Q) a square of the left chart
            c = torch.flip(self.right_chart[0:height, right_min:right_max], dims=[0])
            #
            #dot_temp_mat = ( b[...,None]+c[...,None,:] ).view(height, left_max, batch_size, -1)
            dot_temp_mat = ( b[...,None]+c[...,None,:] ).view(height, left_max, batch_size, self.Q, self.Q)
            # dot temp mat is all_ijdiffs, i-j, batch, Q2

            # add an extra state to the last two dimensions to deal with NULL
            QUASI_INF = 10000000
            null_tensor = torch.tensor([-QUASI_INF])
            null_col_1 = null_tensor.reshape(1, 1, 1, 1, 1).expand(height, left_max, batch_size, self.Q, 1)
            dot_temp_mat = torch.cat([dot_temp_mat, null_col_1], dim=4)
            # dot_temp_mat is now height x left_max x batch_size x Q x (Q+1)
            null_col_2 = null_tensor.reshape(1, 1, 1, 1, 1).expand(height, left_max, batch_size, 1, self.Q+1)
            dot_temp_mat = torch.cat([dot_temp_mat, null_col_2], dim=3)
            # dot_temp_mat is now height x left_max x batch_size x (Q+1) x (Q+1)

            scores_l = self.log_G_l.to(self.device).repeat(height, left_max, batch_size, 1, 1)

            scores_r = self.log_G_r.to(self.device).repeat(height, left_max, batch_size, 1, 1)

            l2r_stacked = self.l2r.unsqueeze(dim=1).repeat(
                height, left_max, batch_size, 1, 1
            ).to(self.device)

            r2l_stacked = self.r2l.unsqueeze(dim=0).repeat(
                height, left_max, batch_size, 1, 1
            ).to(self.device)


            # height x left_max x batch_size x Q
            # throw out the last column for NULL
            l_functor_prob = torch.gather(dot_temp_mat, dim=4, index=l2r_stacked).squeeze(dim=4)[..., :self.Q]
            # height x left_max x batch_size x Q
            r_functor_prob = torch.gather(dot_temp_mat, dim=3, index=r2l_stacked).squeeze(dim=3)[..., :self.Q]

            lscores = scores_l + l_functor_prob[...,None,:]
            rscores = scores_r + r_functor_prob[...,None,:]

            # permute the dims to get i-j, batch, Q, height Q
            lscores = lscores.permute(1,2,3,0,4).contiguous().view(
                left_max, batch_size, self.Q, -1
            )

            rscores = rscores.permute(1,2,3,0,4).contiguous().view(
                left_max, batch_size, self.Q, -1
            )
            lmax_kbc, largmax_kbc = torch.max(lscores, dim=3)
            rmax_kbc, rargmax_kbc = torch.max(rscores, dim=3)
            # i-j, batch, Q


            l_ks = largmax_kbc // (self.Q) + torch.arange(1, left_max+1)[:, None, None]. to(self.device)
            l_bs = largmax_kbc % self.Q
            l2r_repeat = self.l2r.repeat(
                left_max, batch_size, 1
            ).to(self.device)
            l_cs = torch.gather(l2r_repeat, index=l_bs, dim=2)
            l_kbc = torch.stack([l_ks, l_bs, l_cs], dim=0)

            r_ks = rargmax_kbc // (self.Q) + torch.arange(1, left_max+1)[:, None, None]. to(self.device)
            # c is right child, functor in this case
            r_cs = rargmax_kbc % self.Q
            r2l_repeat = self.r2l.repeat(
                left_max, batch_size, 1
            ).to(self.device)
            r_bs = torch.gather(r2l_repeat, index=r_cs, dim=2)
            r_kbc = torch.stack([r_ks, r_bs, r_cs], dim=0)
            lr_kbc = torch.stack([l_kbc, r_kbc], dim=0)


            lr_max = torch.stack([lmax_kbc, rmax_kbc], dim=0)
            combined_max, combined_argmax = torch.max(lr_max, dim=0)

            self.left_chart[height, left_min:left_max] = combined_max
            self.right_chart[height, right_min:right_max] = combined_max

            # repeat to gather the same k, b, and c
            combined_argmax = combined_argmax.repeat(3, 1, 1, 1).unsqueeze(dim=0)
            # TODO use different arrangement of dimensions initally to
            # avoid need for permute
            best_kbc = torch.gather(lr_kbc, index=combined_argmax, dim=0).squeeze(dim=0)
            best_kbc = best_kbc.permute(1, 2, 3, 0)

            backtrack_chart[ij_diff] = best_kbc
        self.right_chart = None
        return backtrack_chart

    # @profile
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
#            if b == self.Q - 1:
#                b_str = "NULL"
#            else:
#                b_str = str(self.ix2cat[b])
#            if c == self.Q - 1:
#                c_str = "NULL"
#            else:
#                c_str = str(self.ix2cat[c])
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

    @staticmethod
    def calc_Q(K=0, D=0):
        if D == -1:
            return K
        return (D+1)*(K)*2

    def find_max_prob_words(self, num_words=10):
        # with torch.no_grad():
        #     embs = self.log_lexis.unsqueeze(-2)
        #     lexis_probs = self.log_lexis.log_prob(embs)
        #     _, max_cats = torch.topk(lexis_probs, num_words, dim=0)
        #     return max_cats.detach().t()
        with torch.no_grad():
            word_indices, prob_vals = zip(*self.vocab_prob_list)
            word_indices, prob_vals = torch.tensor(word_indices).to(self.device), torch.stack(prob_vals, dim=0)
            # all_data = torch.cat([word_indices.unsqueeze(1), prob_vals], dim=1)
            best_word_for_cat = {}

            max_probs, max_indices = torch.topk(prob_vals, num_words, dim=0)
            for cat in range(prob_vals.shape[1]):
                best_word_for_cat[cat] = torch.stack((word_indices[max_indices[:,cat]].float(), max_probs[:,cat]), dim=1)
            return best_word_for_cat

    def _find_max_prob_words_within_vit(self, sents, left_chart_bottom_row):
        with torch.no_grad():
            bottom_row = left_chart_bottom_row.transpose(1, 0) # batch, sentlen, p
            flatten_sents = torch.flatten(sents) # batch*sentlen
            flatten_bottom = torch.flatten(bottom_row, end_dim=-2) # batch*sentlen, p
            # vals, indices = torch.max(flatten_bottom, dim=1) # batch*sentlen, max_p
            for word_index, word in enumerate(flatten_sents):
                raw_word = word.item()
                if raw_word not in self.finished_vocab:
                    self.finished_vocab.add(raw_word)
                    self.vocab_prob_list.append((raw_word, flatten_bottom[word_index].detach()))

    def clear_vocab_prob_list(self):
        self.vocab_prob_list = []
        self.finished_vocab = set()
