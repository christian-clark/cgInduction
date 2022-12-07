import torch, bidict, random, numpy as np, torch.nn.functional as F
from torch import nn
from cky_parser_sgd import BatchCKYParser
from char_coding_models import CharProbRNN, ResidualLayer, \
    WordProbFCFixVocabCompound
from cg_type import CGNode, generate_categories, trees_from_json


class BasicCGInducer(nn.Module):
    def __init__(
            self,
            num_primitives=4,
            max_cat_depth=2,
            cats_json=None,
            state_dim=64,
            num_chars=100,
            device='cpu',
            eval_device="cpu",
            model_type='char',
            num_words=100,
            char_grams_lexicon=None,
            all_words_char_features=None,
            rnn_hidden_dim=320
        ):
        super(BasicCGInducer, self).__init__()
        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.model_type = model_type
        self.num_primitives = num_primitives
        self.max_cat_depth = max_cat_depth
        self.cats_json = cats_json
        self.device = device
        self.eval_device = eval_device

        if self.model_type == 'char':
            self.emit_prob_model = CharProbRNN(
                num_chars,
                state_dim=self.num_cats,
                hidden_size=rnn_hidden_dim
            )
        elif self.model_type == 'word':
            self.emit_prob_model = WordProbFCFixVocabCompound(
                num_words, state_dim
            )
        else:
            raise ValueError("Model type should be char or word")

        self.init_cats_and_masks()
        # CG: "embeddings" for the categories are just one-hot vectors
        # these are used for parent categories, which can only be non-
        # functor (nf) categories
        self.fake_emb = nn.Parameter(torch.eye(self.num_nf_cats))
        # actual embeddings are used to calculate split scores
        # (i.e. prob of terminal vs nonterminal)
        self.nt_emb = nn.Parameter(torch.randn(self.num_cats, state_dim))
        # maps parent_cat to arg_categories x {arg_on_L, arg_on_R}
        self.rule_mlp = nn.Linear(self.num_nf_cats, 2*self.num_nf_cats)

        # example of manually seeding weights (for troubleshooting)
        # note: weight dims are (out_feats, in_feats)
#        QUASI_INF = 10000000.
#        fake_weights = torch.full((17, 8), fill_value=-QUASI_INF)
#        # favor V-aN -> V-aN-bN N
#        fake_weights[1, 3] = 0.
#        # favor V -> N V-aN
#        fake_weights[10, 7] = 0.
#        self.rule_mlp.weight = nn.Parameter(fake_weights)

        self.root_emb = nn.Parameter(torch.eye(1)).to(self.device)
        self.root_mlp = nn.Linear(1, self.num_cats).to(self.device)

        # decides terminal or nonterminal
        self.split_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            nn.Linear(state_dim, 2)
        ).to(self.device)

        self.parser = BatchCKYParser(
            self.ix2cat,
            self.l_functors,
            self.r_functors,
            q=self.num_cats,
            q_nf=self.num_nf_cats,
            device=self.device
        )


    def init_cats_and_masks(self):
        if self.cats_json is not None:
            # TODO implement this option -- will require a different
            # approach to figuring out nf_cats etc
            raise NotImplementedError()
            #all_cats = cat_trees_from_json(self.cats_json)
        else:
            max_depth = self.max_cat_depth
            # TODO pass in optional 3rd arg (max cat depth) from
            # config option
            cats_by_max_depth, ix2cat = generate_categories(
                self.num_primitives,
                self.max_cat_depth
            )

        all_cats = cats_by_max_depth[self.max_cat_depth]
        num_cats = len(all_cats)

        # nf_cats (non-functor cats) are the categories that can be either an 
        # argument
        # taken by a functor category, or the result returned by the
        # functor. Since a functor's depth is one greater than its
        # argument and result, the max depth of an argument or result
        # is self.max_cat_depth-1
        nf_cats = cats_by_max_depth[self.max_cat_depth-1]
        num_nf_cats = len(nf_cats)

        # only allow primitive categories to be at the root of the parse
        # tree
        can_be_root = torch.full((num_cats,), fill_value=-np.inf)
        for cat in cats_by_max_depth[0]:
            assert cat.is_primitive()
            ix = ix2cat.inverse[cat]
            can_be_root[ix] = 0

        # given an argument cat index (i) and a result cat 
        # index (j), l_functors[i, j] gives the functor cat that
        # takes cat i as a right argument and returns cat j 
        # e.g. for the rule V -> V-bN N:
        # l_functors[N, V] = V-bN
        l_functors = torch.empty(
            num_nf_cats, num_nf_cats, dtype=torch.int64
        )

        # same idea but functor appears on the right
        # e.g. for the rule V -> N V-aN:
        # r_functors[N, V] = V-aN
        r_functors = torch.empty(
            num_nf_cats, num_nf_cats, dtype=torch.int64
        )

        for res in nf_cats:
            res_ix = ix2cat.inverse[res]
            for arg in nf_cats:
                arg_ix = ix2cat.inverse[arg]
                # TODO possible not to hardcode operator?
                l_func = CGNode("-b", res, arg)
                l_func_ix = ix2cat.inverse[l_func]
                l_functors[arg_ix, res_ix] = l_func_ix
   
                r_func = CGNode("-a", res, arg)
                r_func_ix = ix2cat.inverse[r_func]
                r_functors[arg_ix, res_ix] = r_func_ix


        print("CEC num cats: {}".format(num_cats))
        self.num_cats = num_cats
        print("CEC num non-functor cats: {}".format(num_nf_cats))
        self.num_nf_cats = num_nf_cats
        self.ix2cat = ix2cat
        print("CEC ix2cat sample: {}".format(random.sample(ix2cat.values(), 100)))
        self.l_functors = l_functors.to(self.device)
        self.r_functors = r_functors.to(self.device)
        self.root_mask = can_be_root.to(self.device)

        
    def forward(self, x, eval=False, argmax=False, use_mean=False, indices=None, set_grammar=True, return_ll=True, **kwargs):
        # x : batch x n
        if set_grammar:
            self.emission = None

            fake_emb = self.fake_emb
            num_cats = self.num_cats
            num_nf_cats = self.num_nf_cats

            # dim: Q
            root_scores = F.log_softmax(
                self.root_mask+self.root_mlp(self.root_emb).squeeze(), dim=0
            )
            full_p0 = root_scores


            # dim: Qnf x 2Qnf
            rule_scores = F.log_softmax(self.rule_mlp(fake_emb), dim=1)
            # dim: Qnf x Qnf
            rule_scores_larg = rule_scores[:, :num_nf_cats]
            # dim: Qnf x Qnf
            rule_scores_rarg = rule_scores[:, num_nf_cats:]

            nt_emb = self.nt_emb
            # dim: Q x 2
            # split_scores[:, 0] gives P(terminal=0 | cat)
            # split_scores[:, 1] gives P(terminal=1 | cat)
            split_scores = F.log_softmax(self.split_mlp(nt_emb), dim=1)

            #full_G_larg = rule_scores_larg + split_scores[:, 0][..., None]
            #full_G_rarg = rule_scores_rarg + split_scores[:, 0][..., None]
            # dim: Qnf x Qnf
            full_G_larg = rule_scores_larg \
                          + split_scores[:num_nf_cats, 0][..., None]
            full_G_rarg = rule_scores_rarg \
                          + split_scores[:num_nf_cats, 0][..., None]

            self.parser.set_models(
                full_p0,
                full_G_larg,
                full_G_rarg,
                self.emission,
                pcfg_split=split_scores
            )


        if self.model_type == 'word':
            x = self.emit_prob_model(x, self.nt_emb, set_grammar=set_grammar)
        else:
            assert self.model_type == "char"
            # TODO better to pass in actual embedding instead of one-hot?
            x = self.emit_prob_model(x, self.fake_emb, set_grammar=set_grammar)

        if argmax:
            if eval and self.device != self.eval_device:
                print("Moving model to {}".format(self.eval_device))
                self.parser.device = self.eval_device
            with torch.no_grad():
                logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list = \
                    self.parser.marginal(x, viterbi_flag=True, only_viterbi=not return_ll, sent_indices=indices)
            if eval and self.device != self.eval_device:
                self.parser.device = self.device
                print("Moving model back to {}".format(self.device))
            return logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list
        else:
            logprob_list, _, _, _ = self.parser.marginal(x)
            logprob_list = logprob_list * (-1)
            return logprob_list
