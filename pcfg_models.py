import torch, bidict, numpy as np
from torch import nn
import torch.nn.functional as F
from cky_parser_sgd import batch_CKY_parser, SMALL_NEGATIVE_NUMBER
from treenode import convert_binary_matrix_to_strtree
from char_coding_models import CharProbRNN, WordProbFCFixVocab, CharProbRNNCategorySpecific, ResidualLayer, WordProbFCFixVocabCompound, CharProbLogistic
from cg_type import enumerate_structures, generate_labeled_trees


class SimpleCompPCFGCharNoDistinction(nn.Module):
    def __init__(self,
                 num_primitives=4,
                 max_cat_depth=2,
                 state_dim=64,
                 num_chars=100,
                 device='cpu',
                 eval_device="cpu",
                 model_type='char',
                 num_words=100,
                 char_grams_lexicon=None,
                 all_words_char_features=None,
                 rnn_hidden_dim=320):
        super(SimpleCompPCFGCharNoDistinction, self).__init__()
        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.model_type = model_type
        self.num_primitives = num_primitives
        self.max_cat_depth = max_cat_depth
        # dummy input matrix for grammar rules

        #self.nont_emb = nn.Parameter(torch.randn(self.all_states, state_dim))
            

        #self.rule_mlp = nn.Linear(state_dim, self.all_states ** 2)
        #self.root_emb = nn.Parameter(torch.randn(1, state_dim))
#        self.root_mlp = nn.Sequential(nn.Linear(state_dim, state_dim),
#                                      ResidualLayer(state_dim, state_dim),
#                                      ResidualLayer(state_dim, state_dim),
#                                      nn.Linear(state_dim, self.all_states))

        # build the matrix for rule weights (C1 -> C2 C3),
        # and the matrix for root weights (probability of each primitive
        # cat being at the root of a tree
        self.initialize_weight_matrices()
        # category embeddings. used for lexical expansion binary branching
        # models

        if self.model_type == 'char':
            self.emit_prob_model = CharProbRNN(num_chars, state_dim=self.num_cats, hidden_size=rnn_hidden_dim)
        elif self.model_type == 'word':
            self.emit_prob_model = WordProbFCFixVocabCompound(num_words, state_dim)
        elif self.model_type == 'subgrams':
            self.emit_prob_model = CharProbLogistic(char_grams_lexicon, all_words_char_features, num_t=self.all_states)

        # CG: "embeddings" for the categories are just one-hot vectors
        self.nont_emb = nn.Parameter(torch.eye(self.num_cats))
        #self.rule_mlp = nn.Linear(self.num_cats, self.num_cats*2)
        self.rule_mlp_l = nn.Linear(self.num_cats, self.num_cats)
        self.rule_mlp_r = nn.Linear(self.num_cats, self.num_cats)
        # assigns 0 weight to any rule that isn't possible under
        # CG constraints
        #self.rule_mlp.weight = self.rule_mlp_weights

        self.root_emb = nn.Parameter(torch.eye(1))
        self.root_mlp = nn.Linear(1, self.num_cats)
        # assigns 0 weight to any non-primitive root candidate
        #self.root_mlp.weight = self.root_mlp_weights

        self.split_mlp = nn.Sequential(nn.Linear(self.num_cats, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       nn.Linear(state_dim, 2))

        self.device = device
        self.eval_device = eval_device
        #self.pcfg_parser = batch_CKY_parser(nt=self.all_states, t=0, device=self.device)
        self.pcfg_parser = batch_CKY_parser(
             self.ix2cat, self.l2r, self.r2l, 
             nt=self.num_cats,
             t=0, device=self.device
        )


    def initialize_weight_matrices(self):
        max_depth = self.max_cat_depth
        nt_options = ["-a", "-b"]
        t_options = list()
        for i in range(self.num_primitives):
            t_options.append(str(i))
        #t_options = ["0", "1", "2", "3"]
        structs = enumerate_structures(max_depth)

        all_trees = list()
        for s in structs:
            lts = generate_labeled_trees(s, nt_options, t_options)
            all_trees.extend(lts)

        ix2cat = bidict.bidict()
        for t in all_trees:
            ix2cat[len(ix2cat)] = t

        # +1 for a special NULL category at the end of the list
        num_cats = len(ix2cat) + 1
        null_cat_ix = num_cats - 1

        #can_be_parent_child = torch.zeros(num_cats, 2, num_cats)
        #can_be_parent_lchild = torch.zeros(num_cats, num_cats)
        #can_be_parent_rchild = torch.zeros(num_cats, num_cats)
        # TODO change to np.inf?
        can_be_parent_lchild = torch.full((num_cats, num_cats), fill_value=-10000000)
        can_be_parent_rchild = torch.full((num_cats, num_cats), fill_value=-10000000)
        # TODO probably don't want these but these make NULL acceptable
        #can_be_parent_lchild[:, null_cat_ix] = 0
        #can_be_parent_rchild[:, null_cat_ix] = 0
        #parent_child_weights = torch.full(
        #    (num_cats, 2, num_cats), fill_value=-np.inf
        #)
        #can_be_root = torch.zeros(num_cats)
        can_be_root = torch.full((num_cats,), fill_value=-np.inf)

        # maps category on left (index) to argument taken this category
        # (value at index). If the category on the left cannot take an
        # argument, the value will be num_cats (a dummy category)
        l2r = torch.empty(num_cats, dtype=torch.int64)
        l2r[null_cat_ix] = null_cat_ix
        # vice versa
        r2l = torch.empty(num_cats, dtype=torch.int64)
        r2l[null_cat_ix] = null_cat_ix
            
        for cat_ix in ix2cat:
            cat = ix2cat[cat_ix]
            kittens = cat.children(cat.root)
            # only primitives cats can be the root of a parse tree
            if len(kittens) == 0:
                can_be_root[cat_ix] = 0
                # primitives can't take arguments, so NULL
                l2r[cat_ix] = null_cat_ix
                r2l[cat_ix] = null_cat_ix
            else:
                assert len(kittens) == 2
                kitten1, kitten2 = kittens
                kitten1 = cat.subtree(kitten1.identifier)
                kitten1_ix = ix2cat.inverse[kitten1]
                kitten2 = cat.subtree(kitten2.identifier)
                kitten2_ix = ix2cat.inverse[kitten2]
                root_tag = cat.get_node(cat.root).tag 
                if root_tag == "-b":
                    # left child is functor
                    can_be_parent_lchild[kitten1_ix, cat_ix] = 0
                    l2r[cat_ix] = kitten2_ix
                    r2l[cat_ix] = null_cat_ix
                else:
                    assert root_tag == "-a"
                    # right child is functor
                    can_be_parent_rchild[kitten1_ix, cat_ix] = 0
                    r2l[cat_ix] = kitten2_ix
                    l2r[cat_ix] = null_cat_ix


        print("CEC num cats: {}".format(num_cats))
        self.num_cats = num_cats
        self.ix2cat = ix2cat
        self.l2r = l2r
        self.r2l = r2l
        self.rule_filter_l = can_be_parent_lchild
        self.rule_filter_r = can_be_parent_rchild
        # TODO does bias matter for either of these?
        #self.rule_weights = nn.Parameter(
        #    parent_child_weights.reshape(num_cats, 2*num_cats).t()
        #)
        self.root_filter = can_be_root
        #self.root_mlp_weights = nn.Parameter(
        #    can_be_root.unsqueeze(dim=1)
        #)


    def forward(self, x, eval=False, argmax=False, use_mean=False, indices=None, set_pcfg=True, return_ll=True, **kwargs):
        # x : batch x n
        if set_pcfg:
            self.emission = None

            nt_emb = self.nont_emb

            #root_scores = F.log_softmax(self.root_mlp(self.root_emb).squeeze(), dim=0)
            root_scores = F.log_softmax(
                self.root_filter+self.root_mlp(self.root_emb).squeeze(), dim=0
            )
            full_p0 = root_scores
            #rule_score = F.log_softmax(self.rule_mlp(nt_emb), dim=1)  # nt x t**2
            #rule_score = F.softmax(self.rule_mlp(nt_emb), dim=1)  # nt x t**2

            # TODO add in rule filter again
            #rule_score_l = F.log_softmax(
            #    self.rule_mlp_l(nt_emb), dim=1
            #)  # nt x t**2
            #rule_score_l = F.log_softmax(
            #    self.rule_filter_l+self.rule_mlp_l(nt_emb), dim=1
            #)  # nt x t**2
            rule_score_l = F.log_softmax(
                self.rule_mlp_l(nt_emb), dim=1
            ) + self.rule_filter_l  # nt x t**2

            # TODO readd this maybe?
            # set prob of NULL children to 0 after softmax
            #rule_score_l[:, self.num_cats-1] = -float('inf')

            #print("CEC rule filter l:")
            #print(self.rule_filter_l)
            #print("CEC sum l:")
            #print(self.rule_filter_l+self.rule_mlp_l(nt_emb))
            #print("CEC rule score l:")
            #print(rule_score_l)

            # TODO add in rule filter again
            #rule_score_r = F.log_softmax(
            #    self.rule_mlp_r(nt_emb), dim=1
            #)  # nt x t**2
            #rule_score_r = F.log_softmax(
            #    self.rule_filter_r+self.rule_mlp_r(nt_emb), dim=1
            #)  # nt x t**2
            rule_score_r = F.log_softmax(
                self.rule_mlp_r(nt_emb), dim=1
            ) + self.rule_filter_r  # nt x t**2

            # TODO readd this maybe?
            #rule_score_r[:, self.num_cats-1] = -float('inf')

            #print("CEC rule score r:")
            #print(rule_score_r)
            #full_G = rule_score
            # split_scores[:, 0] gives P(terminal=0 | cat)
            # split_scores[:, 1] gives P(terminal=1 | cat)
            #split_scores = F.log_softmax(self.split_mlp(nt_emb), dim=1)
            split_scores = F.log_softmax(self.split_mlp(nt_emb), dim=1)
            split_scores = F.log_softmax(self.split_mlp(nt_emb), dim=1)
            #full_G = full_G + split_scores[:, 0][..., None]
            full_G_l = rule_score_l + split_scores[:, 0][..., None]
            full_G_r = rule_score_r + split_scores[:, 0][..., None]

            self.pcfg_parser.set_models(full_p0, full_G_l, full_G_r, self.emission, pcfg_split=split_scores)

        if self.model_type != 'subgrams':
            # TODO better to pass in actual embedding instead of one-hot?
            x = self.emit_prob_model(x, self.nont_emb, set_pcfg=set_pcfg)
        else:
            x = self.emit_prob_model(x)

        if argmax:
            if eval and self.device != self.eval_device:
                print("Moving model to {}".format(self.eval_device))
                self.pcfg_parser.device = self.eval_device
            with torch.no_grad():
                logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list = \
                    self.pcfg_parser.marginal(x, viterbi_flag=True, only_viterbi=not return_ll, sent_indices=indices)
            if eval and self.device != self.eval_device:
                self.pcfg_parser.device = self.device
                print("Moving model back to {}".format(self.device))
            return logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list
        else:
            logprob_list, _, _, _ = self.pcfg_parser.marginal(x)
            logprob_list = logprob_list * (-1)
            return logprob_list
