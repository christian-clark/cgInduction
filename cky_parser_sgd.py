import torch
from treenode import Node, nodes_to_tree


SMALL_NEGATIVE_NUMBER = -1e8
DEBUG = False

def printDebug(*args, **kwargs):
    if DEBUG:
        print("DEBUG: ", end="")
        print(*args, **kwargs)


class BatchCKYParser:
    def __init__(
        self, ix2cat, ix2cat_gen, lfunc_ixs, rfunc_ixs, qall, qgen,
        device="cpu"
    ):
        # TODO figure out what D and K do. K might not be set correctly
        self.D = -1
        self.K = qall
        self.lexis = None
        self.ix2cat = ix2cat
        self.ix2cat_gen = ix2cat_gen
        self.lfunc_ixs = lfunc_ixs
        self.rfunc_ixs = rfunc_ixs
        self.qall = qall
        self.qgen = qgen
        self.curr_sent_len = -1
        self.counter = 0
        self.vocab_prob_list = []
        self.finished_vocab = set()
        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'


    def set_models(
            self, root_probs, full_G, split_probs=None
        ):
        self.root_probs = root_probs
        self.full_G = full_G
        self.split_probs = split_probs
        # TODO remove?
        self.log_lexis = None


    def marginal(self, sents, viterbi_flag=False, only_viterbi=False, sent_indices=None):
        self.sent_indices = sent_indices

        if not only_viterbi:
            self.compute_inside_logspace(sents)
            # nodes_list, logprob_list = self.sample_tree(sents)
            # nodes_list, logprob_list = self.sample_tree_logspace(sents)
            logprob_list = self.marginal_likelihood_logspace()
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
                    this_vnodes = self.viterbi_backtrack(backchart)
                    vnodes += this_vnodes

        self.curr_sent_len = -1

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
            self.curr_sent_len = len(sents[0])
        except:
            print(sents)
            raise
        batch_size = len(sents)
        sent_len = self.curr_sent_len

        # left chart is the left right triangle of the chart, the top row is the lexical items, and the bottom cell is the
        #  top cell. The right chart is the left chart pushed against the right edge of the chart. The chart is a square.
        # dim: sentlen x sentlen x batch x qall
        self.left_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.qall)
        ).float().to(self.device)
        # dim: sentlen x sentlen x batch x qall
        self.right_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.qall)
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
            # dim: height x imax x batch_size x qall
            b = self.left_chart[0:height, imin:imax]
            # dim: height x imax x batch_size x qall
            c = torch.flip(self.right_chart[0:height, jmin:jmax], dims=[0])

            # indices for cats that can be generated children
            gen_ixs = torch.tensor(
                [self.ix2cat.inv[c] for c in self.ix2cat_gen.values()]
            ).to(self.device)
            # dim: height x imax x batch_size x qgen
            gen_ixs = gen_ixs.repeat(
                height, imax, batch_size, 1
            )

            # dim: height x imax x batch_size x qgen
            # torch throws an error about inplace modification if the clone()
            # isn't there...idk why
            b_gen = b.clone().gather(dim=-1, index=gen_ixs)
            # probability of generated i on the left followed by implicit j
            # on the right
            # NOTE: doing the logsumexp here means this takes more memory
            # than the parallel line in compute_viterbi_inside
            # dim: imax x batch_size x qall x qgen
            children_prob_lgen = torch.logsumexp(
                b_gen[...,None,:] + c[...,None], dim=0
            )

            # dim: height x imax x batch_size x qgen
            c_gen = c.gather(dim=-1, index=gen_ixs)
            # probability of implicit i on the left followed by generated j
            # on the right
            # dim: imax x batch_size x qall x qgen
            children_prob_rgen = torch.logsumexp(
                b[...,None] + c_gen[...,None,:], dim=0
            )

            # probability that parent category i branches into left argument j
            # and right functor i-aj
            # dim: imax x batch_size x qall x qgen
            G_Aa = self.full_G[:, :self.qgen].to(self.device).repeat(
                imax, batch_size, 1, 1
            )
            # probability that parent category i branches into left functor i-bj
            # and right argument j
            # dim:  imax x batch_size x qall x qgen
            G_Ab = self.full_G[:, self.qgen:2*self.qgen].to(self.device).repeat(
                imax, batch_size, 1, 1
            )
            # probability that parent category i branches into left modifier j
            # and right modificand i
            # dim: imax x batch_size x qall x qgen
            G_Ma = self.full_G[:, 2*self.qgen:3*self.qgen].to(self.device).repeat(
                imax, batch_size, 1, 1
            )
            # probability that parent category i branches into left
            # modificand i and right modifier j
            # dim:  imax x batch_size x qall x qgen
            G_Mb = self.full_G[:, 3*self.qgen:].to(self.device).repeat(
                imax, batch_size, 1, 1
            )
        
            # rfunc_ixs[..., i, j] tells the index of the right-child functor
            # going with parent cat i, left argument cat j
            # dim: imax x batch_size x qall x qgen
            rfunc_ixs = self.rfunc_ixs.repeat(imax, batch_size, 1, 1)
            # lfunc_ixs[..., i, j] tells the index of the left-child functor
            # going with parent cat i, right argument cat j
            # dim: imax x batch_size x qall x qgen
            lfunc_ixs = self.lfunc_ixs.repeat(imax, batch_size, 1, 1)

            # rearrange children_prob_lgen to index by parent
            # and argument rather than functor and argument
            # dim: imax x batch_size x qall x qgen
            children_prob_Aa = torch.gather(
                children_prob_lgen, dim=2, index=rfunc_ixs
            )

            # rearrange children_prob_rgen to index by parent
            # and argument rather than functor and argument
            # dim: imax x batch_size x qall x qgen
            children_prob_Ab = torch.gather(
                children_prob_rgen, dim=2, index=lfunc_ixs
            )

            # dim: imax x batch_size x qall x qgen
            combined_Aa = G_Aa + children_prob_Aa
            combined_Ab = G_Ab + children_prob_Ab
            # NOTE: because the implicit child and parent have the same
            # predcat for modifier attachment, there's no need to define
            # children_prob_Ma and children_prob_Mb
            combined_Ma = G_Ma + children_prob_lgen
            combined_Mb = G_Mb + children_prob_rgen

            # dim: 4 x imax x batchsize x qall x qgen
            combined = torch.stack(
                [combined_Aa, combined_Ab, combined_Ma, combined_Mb]
            )
            # combine probabilities across operations (dim 0) and gen
            # categories (dim 3)
            # dim: imax x batchsize x qall
            combined = combined.logsumexp(dim=0).logsumexp(dim=3)
            # TODO does combined need to be expanded somehow? cf what happens
            # to y1 in the head_predicate git branch
            self.left_chart[height, imin:imax] = combined
            self.right_chart[height, jmin:jmax] = combined


    def marginal_likelihood_logspace(self):
        sent_len = self.curr_sent_len
        # dim: batch_size x qall
        topnode_pdf = self.left_chart[sent_len-1, 0]
        # dim: batch_size x qall
        p_topnode = topnode_pdf + self.root_probs
        # dim: batch_size
        logprobs = torch.logsumexp(p_topnode, dim=1)
        return logprobs


    def compute_viterbi_inside(self, sent):
        self.curr_sent_len = sent.shape[1]
        sent_len = self.curr_sent_len
        batch_size = 1
        self.left_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.qall)
        ).float().to(self.device)
        self.right_chart = torch.zeros(
            (sent_len, sent_len, batch_size, self.qall)
        ).float().to(self.device)
        backtrack_chart = {}
        self.get_lexis_prob(sent, self.left_chart)
        self.right_chart[0] = self.left_chart[0]

        for ij_diff in range(1, sent_len):
            printDebug("ij_diff: {}".format(ij_diff))
            imin = 0
            imax = sent_len - ij_diff
            jmin = ij_diff
            jmax = sent_len
            height = ij_diff

            # a square of the left chart
            # dim: height x imax x batch_size x qall
            b = self.left_chart[0:height, imin:imax]
            # dim: height x imax x batch_size x qall
            c = torch.flip(self.right_chart[0:height, jmin:jmax], dims=[0])

            # indices for cats that can be generated children
            gen_ixs = torch.tensor(
                [self.ix2cat.inv[c] for c in self.ix2cat_gen.values()]
            ).to(self.device)
            # dim: height x imax x batch_size x qgen
            gen_ixs = gen_ixs.repeat(
                height, imax, batch_size, 1
            )

            # dim: height x imax x batch_size x qgen
            # torch throws an error about inplace modification if the clone()
            # isn't there...idk why
            b_gen = b.clone().gather(dim=-1, index=gen_ixs)
            # probability of generated i on the left followed by implicit j
            # on the right
            # NOTE: doing the logsumexp here means this takes more memory
            # than the parallel line in compute_viterbi_inside
            # dim: height x imax x batch_size x qall x qgen
            children_prob_lgen = b_gen[...,None,:] + c[...,None]

            # dim: height x imax x batch_size x qgen
            c_gen = c.gather(dim=-1, index=gen_ixs)
            # probability of implicit i on the left followed by generated j
            # on the right
            # dim: height x imax x batch_size x qall x qgen
            children_prob_rgen = b[...,None] + c_gen[...,None,:]

            # probability that parent category i branches into left argument j
            # and right functor i-aj
            # dim: height x imax x batch_size x qall x qgen
            G_Aa = self.full_G[:, :self.qgen].to(self.device).repeat(
                height, imax, batch_size, 1, 1
            )
            # probability that parent category i branches into left functor i-bj
            # and right argument j
            # dim: height x imax x batch_size x qall x qgen
            G_Ab = self.full_G[:, self.qgen:2*self.qgen].to(self.device).repeat(
                height, imax, batch_size, 1, 1
            )
            # probability that parent category i branches into left modifier j
            # and right modificand i
            # dim: height x imax x batch_size x qall x qgen
            G_Ma = self.full_G[:, 2*self.qgen:3*self.qgen].to(self.device).repeat(
                height, imax, batch_size, 1, 1
            )
            # probability that parent category i branches into left
            # modificand i and right modifier j
            # dim: height x imax x batch_size x qall x qgen
            G_Mb = self.full_G[:, 3*self.qgen:].to(self.device).repeat(
                height, imax, batch_size, 1, 1
            )

            # rfunc_ixs[..., i, j] tells the index of the right-child functor
            # going with parent cat i, larg cat j
            # dim: height x imax x batch_size x qres x qarg
            rfunc_ixs = self.rfunc_ixs.repeat(height, imax, batch_size, 1, 1)
            # lfunc_ixs[..., i, j] tells the index of the left-child functor
            # going with parent cat i, rarg cat j
            # dim: imax x batch_size x qres x qarg
            lfunc_ixs = self.lfunc_ixs.repeat(height, imax, batch_size, 1, 1)

            # rearrange children_prob_lgen to index by parent
            # and argument rather than functor and argument
            # dim: height x imax x batch_size x qall x qgen
            children_prob_Aa = torch.gather(
                children_prob_lgen, dim=3, index=rfunc_ixs
            )

            # rearrange children_prob_rgen to index by parent
            # and argument rather than functor and argument
            # dim: height x imax x batch_size x qall x qgen
            children_prob_Ab = torch.gather(
                children_prob_rgen, dim=3, index=lfunc_ixs
            )

            ################### Best kbc for Aa operation
            # probability that parent category i branches into left argument j
            # and right functor i-aj, that category j spans the words on the
            # left, and that category i-aj spans the words on the right
            # dim: height x imax x batch_size x qall x qgen
            combined_Aa = G_Aa + children_prob_Aa
            combined_Aa = combined_Aa.permute(1,2,3,0,4)
            # dim: imax x batch_size x qall x height*qgen
            combined_Aa = combined_Aa.contiguous().view(
                imax, batch_size, self.qall, -1
            )

            # dim: imax x batch_size x qall
            max_kbc_Aa, argmax_kbc_Aa = torch.max(combined_Aa, dim=3)

            # dim: imax x batch_size x qall
            ks_Aa = torch.div(argmax_kbc_Aa, self.Qgen, rounding_mode="floor") \
                   + torch.arange(1, imax+1)[:, None, None]. to(self.device)

            # NOTE: these are the predcat indices based on the indexing for
            # argument predcats
            # dim: imax x batch_size x qall
            bs_Aa = argmax_kbc_Aa % (self.qgen)

            # dim: imax x batch_size x qall x 1
            bs_reshape_Aa = bs_Aa.view(imax, batch_size, self.qall, 1)

            # dim: imax x batch_size x qall
            rfunc_ixs = rfunc_ixs[0]
            cs_Aa = torch.gather(rfunc_ixs, index=bs_reshape_Aa, dim=3).squeeze(dim=3)

            # maps index of an gen cat to its index in the full set of cats.
            gencat2cat = [
                self.ix2cat.inv[self.ix2cat_gen[i]] \
                for i in range(len(self.ix2cat_gen))
            ]
            gencat2cat = torch.tensor(gencat2cat).to(self.device)

            # dim: imax x batch_size x qgen
            cat_ix = gencat2cat.repeat(imax, batch_size, 1)

            # dim: imax x batch_size x qall
            # now each entry is an index for ix2cat instead of
            # ix2cat_gen. This is necessary for so that l_cs and l_bs
            # use the same indexing for viterbi_backtrack
            bs_reindexed_Aa = torch.gather(cat_ix, dim=-1, index=bs_Aa)

            # dim: 3 x imax x batch_size x par
            kbc_Aa = torch.stack([ks_Aa, bs_reindexed_Aa, cs_Aa], dim=0)

            ################### Best kbc for Ab operation
            # probability that parent category i branches into left functor
            # i-bj and right argument j, that category i-bj spans the words on
            # the left, and that category j spans the words on the right
            # dim: height x imax x batch_size x qres x qarg
            combined_Ab = G_Ab + children_prob_Ab
            combined_Ab = combined_Ab.permute(1,2,3,0,4)
            # dim: imax x batch_size x qall x height*qgen
            combined_Ab = combined_Ab.contiguous().view(
                imax, batch_size, self.qall, -1
            )

            # dim: imax x batch_size x qall
            max_kbc_Ab, argmax_kbc_Ab = torch.max(combined_Ab, dim=3)

            # dim: imax x batch_size x qall
            ks_Ab = torch.div(argmax_kbc_Ab, self.qgen, rounding_mode="floor") \
                   + torch.arange(1, imax+1)[:, None, None]. to(self.device)
            # dim: imax x batch_size x Qall
            cs_Ab = argmax_kbc_Ab % (self.qgen)

            # dim: imax x batch_size x qall x 1
            cs_reshape_Ab = cs_Ab.view(imax, batch_size, self.qall, 1)

            # dim: imax x batch_size x qall
            lfunc_ixs = lfunc_ixs[0]
            bs_Ab = torch.gather(lfunc_ixs, index=cs_reshape_Ab, dim=3).squeeze(dim=3)
            # dim: imax x batch_size x qall
            cs_reindexed_Ab = torch.gather(cat_ix, dim=-1, index=cs_Ab)

            # dim: 3 x imax x batch_size x qall
            kbc_Ab = torch.stack([ks_Ab, bs_Ab, cs_reindexed_Ab], dim=0)

            ################### Best kbc for Ma operation
            combined_Ma = G_Ma + children_prob_lgen
            combined_Ma = combined_Ma.permute(1,2,3,0,4)
            # dim: imax x batch_size x qall x height*Qgen
            combined_Ma = combined_Ma.contiguous().view(
                imax, batch_size, self.qall, -1
            )

            # dim: imax x batch_size x qall
            max_kbc_Ma, argmax_kbc_Ma = torch.max(combined_Ma, dim=3)

            # dim: imax x batch_size x qall
            ks_Ma = torch.div(argmax_kbc_Ma, self.qgen, rounding_mode="floor") \
                   + torch.arange(1, imax+1)[:, None, None]. to(self.device)

            # dim: imax x batch_size x qall
            bs_Ma = argmax_kbc_Ma % (self.qgen)

            # dim: imax x batch_size x qall
            # don't have to reorganize cs because parent and implicit child are
            # the same for modification
            # TODO make sure this is correct
            cs_Ma = torch.arange(self.qall).to(self.device)
            cs_Ma = cs_Ma.unsqueeze(dim=0).unsqueeze(dim=0).repeat(imax, batch_size, 1)

            # dim: imax x batch_size x qall
            # now each entry is an index for ix2predcat instead of
            # ix2cat_gen. This is necessary for so that l_cs and l_bs
            # use the same indexing for viterbi_backtrack
            bs_reindexed_Ma = torch.gather(cat_ix, dim=-1, index=bs_Ma)

            # dim: 3 x imax x batch_size x par
            kbc_Ma = torch.stack([ks_Ma, bs_reindexed_Ma, cs_Ma], dim=0)

            ################### Best kbc for Mb operation
            combined_Mb = G_Mb + children_prob_rgen
            combined_Mb = combined_Mb.permute(1,2,3,0,4)
            # dim: imax x batch_size x qall x height*qgen
            combined_Mb = combined_Mb.contiguous().view(
                imax, batch_size, self.qall, -1
            )

            # dim: imax x batch_size x qall
            max_kbc_Mb, argmax_kbc_Mb = torch.max(combined_Mb, dim=3)

            # dim: imax x batch_size x qall
            ks_Mb = torch.div(argmax_kbc_Mb, self.qgen, rounding_mode="floor") \
                   + torch.arange(1, imax+1)[:, None, None]. to(self.device)
            # dim: imax x batch_size x qall
            cs_Mb = argmax_kbc_Mb % (self.qgen)

            # dim: imax x batch_size x qall
            # don't have to reorganize cs because parent and implicit child are
            # the same for modification
            # TODO make sure this is correct
            bs_Mb = torch.arange(self.qall).to(self.device)
            bs_Mb = bs_Mb.unsqueeze(dim=0).unsqueeze(dim=0).repeat(imax, batch_size, 1)

            # dim: imax x batch_size x qall
            cs_reindexed_Mb = torch.gather(cat_ix, dim=-1, index=cs_Mb)

            # dim: 3 x imax x batch_size x qall
            kbc_Mb = torch.stack([ks_Mb, bs_Mb, cs_reindexed_Mb], dim=0)

            ################### stack kbcs and find the very best

            # dim: 2 x 3 x imax x batch_size x qall
            kbc_allOp = torch.stack([kbc_Aa, kbc_Ab, kbc_Ma, kbc_Mb], dim=0)

            # dim: 2 x imax x batch_size x qall
            max_allOp = torch.stack(
                [max_kbc_Aa, max_kbc_Ab, max_kbc_Ma, max_kbc_Mb], dim=0
            )

            # tells which operation is most likely
            # each value of the argmax is:
            # - 0 (Aa)
            # - 1 (Ab)
            # - 2 (Ma)
            # - 3 (Mb)
            # dim: imax x batch_size x qall
            combined_max, combined_argmax = torch.max(max_allOp, dim=0)

            self.left_chart[height, imin:imax] = combined_max
            self.right_chart[height, jmin:jmax] = combined_max

            # gather k, b, and c
            # dim: 1 x 3 x imax x batch_size x qall
            combined_argmax = combined_argmax.repeat(3, 1, 1, 1).unsqueeze(dim=0)
            # TODO use different arrangement of dimensions initally to
            # avoid need for permute
            best_kbc = torch.gather(kbc_allOp, index=combined_argmax, dim=0)
            # dim: 3 x imax x batch_size x qres
            best_kbc = best_kbc.squeeze(dim=0)
            # dim: imax x batch_size x qres x 3
            best_kbc = best_kbc.permute(1, 2, 3, 0)
            backtrack_chart[ij_diff] = best_kbc
        self.right_chart = None
        return backtrack_chart


    def viterbi_backtrack(self, backtrack_chart, max_cats=None):
        sent_index = 0
        nodes_list = []
        sent_len = self.curr_sent_len
        topnode_pdf = self.left_chart[sent_len-1, 0]
        if max_cats is not None:
            max_cats = max_cats.squeeze()
            max_cats = max_cats.tolist()

        # draw the top node
        p_topnode = topnode_pdf + self.root_probs
        A_ll, top_A = torch.max(p_topnode, dim=-1)
        # top_A = top_A.squeeze()
        # A_ll = A_ll.squeeze()

        expanding_nodes = []
        expanded_nodes = []
        # rules = []
        assert self.curr_sent_len > 0, "must call inside pass first!"

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

