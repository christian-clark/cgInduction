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
            max_func_depth=2,
            max_arg_depth=None,
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
        self.max_func_depth = max_func_depth
        self.max_arg_depth = max_arg_depth
        self.cats_json = cats_json
        self.device = device
        self.eval_device = eval_device

        if self.model_type == 'char':
            self.emit_prob_model = CharProbRNN(
                num_chars,
                state_dim=self.state_dim,
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
        # these are used for result categories
        self.fake_emb = nn.Parameter(torch.eye(self.num_res_cats))
        # actual embeddings are used to calculate split scores
        # (i.e. prob of terminal vs nonterminal)
        self.nt_emb = nn.Parameter(torch.randn(self.num_func_cats, state_dim))
        # maps res_cat to arg_cat x {arg_on_L, arg_on_R}
        self.rule_mlp = nn.Linear(self.num_res_cats, 2*self.num_arg_cats)

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
        self.root_mlp = nn.Linear(1, self.num_func_cats).to(self.device)

        # decides terminal or nonterminal
        self.split_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            nn.Linear(state_dim, 2)
        ).to(self.device)

        self.parser = BatchCKYParser(
            self.ix2cat,
            self.lfunc_ixs,
            self.rfunc_ixs,
            qfunc=self.num_func_cats,
            qres=self.num_res_cats,
            qarg=self.num_arg_cats,
            device=self.device
        )


    def init_cats_and_masks(self):
        if self.cats_json is not None:
            # TODO implement this option -- will require a different
            # approach to figuring out res_cats etc
            raise NotImplementedError()
            #all_cats = cat_trees_from_json(self.cats_json)
        else:
            cats_by_max_depth, ix2cat = generate_categories(
                self.num_primitives,
                self.max_func_depth,
                self.max_arg_depth
            )

        print("CEC len(ix2cat): {}".format(len(ix2cat)))

        func_cats = cats_by_max_depth[self.max_func_depth]
        num_func_cats = len(func_cats)

        # res_cats (result cats) are the categories that can be
        # a result from a functor applying to its argument.
        # Since a functor's depth is one greater than the max depth between
        # its argument and result, the max depth of a result
        # is self.max_func_depth-1
        res_cats = cats_by_max_depth[self.max_func_depth-1]
        num_res_cats = len(res_cats)

        # optionally constrain the complexity of argument categories
        if self.max_arg_depth is None:
            arg_cats = res_cats
            num_arg_cats = num_res_cats
        else:
            arg_cats = cats_by_max_depth[self.max_arg_depth]
            num_arg_cats = len(arg_cats)

        # only allow primitive categories to be at the root of the parse
        # tree
        can_be_root = torch.full((num_func_cats,), fill_value=-np.inf)
        for cat in cats_by_max_depth[0]:
            assert cat.is_primitive()
            ix = ix2cat.inverse[cat]
            can_be_root[ix] = 0

        # given an result cat index (i) and an argument cat 
        # index (j), lfunc_ixs[i, j] gives the functor cat that
        # takes cat j as a right argument and returns cat i
        # e.g. for the rule V -> V-bN N:
        # lfunc_ixs[V, N] = V-bN
        lfunc_ixs = torch.empty(
            num_res_cats, num_arg_cats, dtype=torch.int64
        )

        # same idea but functor appears on the right
        # e.g. for the rule V -> N V-aN:
        # rfunc_ixs[V, N] = V-aN
        rfunc_ixs = torch.empty(
            num_res_cats, num_arg_cats, dtype=torch.int64
        )

        for res in res_cats:
            res_ix = ix2cat.inverse[res]
            for arg in arg_cats:
                arg_ix = ix2cat.inverse[arg]
                # TODO possible not to hardcode operator?
                lfunc = CGNode("-b", res, arg)
                lfunc_ix = ix2cat.inverse[lfunc]
                lfunc_ixs[res_ix, arg_ix] = lfunc_ix
                rfunc = CGNode("-a", res, arg)
                rfunc_ix = ix2cat.inverse[rfunc]
                rfunc_ixs[res_ix, arg_ix] = rfunc_ix


        print("CEC num func cats: {}".format(num_func_cats))
        self.num_func_cats = num_func_cats
        print("CEC num res cats: {}".format(num_res_cats))
        self.num_res_cats = num_res_cats
        print("CEC num arg cats: {}".format(num_arg_cats))
        self.num_arg_cats = num_arg_cats
        self.ix2cat = ix2cat
        #print("CEC ix2cat sample: {}".format(random.sample(ix2cat.items(), 100)))
        print("CEC ix2cat sample: {}".format(list(ix2cat.items())[:100]))
        #print("CEC ix2cat: {}".format(ix2cat))
        self.lfunc_ixs = lfunc_ixs.to(self.device)
        self.rfunc_ixs = rfunc_ixs.to(self.device)
        self.root_mask = can_be_root.to(self.device)

        
    def forward(self, x, eval=False, argmax=False, use_mean=False, indices=None, set_grammar=True, return_ll=True, **kwargs):
        # x : batch x n
        if set_grammar:
            self.emission = None

            fake_emb = self.fake_emb
            num_func_cats = self.num_func_cats
            num_res_cats = self.num_res_cats
            num_arg_cats = self.num_arg_cats

            # dim: Qfunc
            root_scores = F.log_softmax(
                self.root_mask+self.root_mlp(self.root_emb).squeeze(), dim=0
            )
            full_p0 = root_scores

            LARG_PENALTY = 100
            larg_penalty = torch.full((num_res_cats, num_arg_cats), -LARG_PENALTY)
            rarg_penalty = torch.full((num_res_cats, num_arg_cats), 0)
            penalty = torch.concat([larg_penalty, rarg_penalty], dim=1).to(self.device)

            # dim: Qres x 2Qarg
            # use this line to add a bias toward forward function application (i.e.
            # argument always on right)
            #rule_scores = F.log_softmax(self.rule_mlp(fake_emb)+penalty, dim=1)
            rule_scores = F.log_softmax(self.rule_mlp(fake_emb), dim=1)
            # dim: Qres x Qarg
            rule_scores_larg = rule_scores[:, :num_arg_cats]
            # dim: Qres x Qarg
            rule_scores_rarg = rule_scores[:, num_arg_cats:]


            nt_emb = self.nt_emb
            # dim: Qfunc x 2
            # split_scores[:, 0] gives P(terminal=0 | cat)
            # split_scores[:, 1] gives P(terminal=1 | cat)
            #split_scores = F.log_softmax(nn.Dropout()(self.split_mlp(nt_emb)), dim=1)
            split_scores = F.log_softmax(self.split_mlp(nt_emb), dim=1)

            #full_G_larg = rule_scores_larg + split_scores[:, 0][..., None]
            #full_G_rarg = rule_scores_rarg + split_scores[:, 0][..., None]
            # dim: Qres x Qarg
            full_G_larg = rule_scores_larg \
                          + split_scores[:num_res_cats, 0][..., None]
            full_G_rarg = rule_scores_rarg \
                          + split_scores[:num_res_cats, 0][..., None]

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
            x = self.emit_prob_model(x, self.nt_emb, set_grammar=set_grammar)

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
