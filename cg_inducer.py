import torch, bidict, numpy as np, torch.nn.functional as F
from torch import nn
from cky_parser_sgd import BatchCKYParser
from char_coding_models import CharProbRNN, ResidualLayer, \
    WordProbFCFixVocabCompound
from cg_type import enumerate_structures, generate_labeled_trees, \
    trees_from_json


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

        # CG: "embeddings" for the categories are just one-hot vectors
        self.fake_emb = nn.Parameter(torch.eye(self.num_cats))
        # actual embeddings are used to calculate split scores and
        # probs of left vs right functors
        self.nt_emb = nn.Parameter(torch.randn(self.num_cats, state_dim))
        # output dim comes from
        # possible_functor_categories x {functor_on_L, functor_on_R} + NULL
        # NULL is for the case that no available categories could be the
        # functor for the parent category
        self.rule_mlp = nn.Linear(self.num_cats, 2*self.num_cats+1)

        # example of manually seeding weights (for troubleshooting)
        # note: weight dims are (out_feats, in_feats)
#        QUASI_INF = 10000000.
#        fake_weights = torch.full((17, 8), fill_value=-QUASI_INF)
#        # favor V-aN -> V-aN-bN N
#        fake_weights[1, 3] = 0.
#        # favor V -> N V-aN
#        fake_weights[10, 7] = 0.
#        self.rule_mlp.weight = nn.Parameter(fake_weights)

        self.root_emb = nn.Parameter(torch.eye(1))
        self.root_mlp = nn.Linear(1, self.num_cats)

        # decides terminal or nonterminal
        self.split_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            nn.Linear(state_dim, 2)
        )

        self.device = device
        self.eval_device = eval_device
        self.parser = BatchCKYParser(
            self.ix2cat,
            self.l2r,
            self.r2l, 
            nt=self.num_cats,
            t=0,
            device=self.device
        )

            self.init_cats_and_masks()


    def init_cats_and_masks(self):
        if self.cats_json is not None:
            all_cat_trees = cat_trees_from_json(self.cats_json)
        else:
            max_depth = self.max_cat_depth
            # -a and -b are the same as \ and / in other notation
            operations = ["-a", "-b"]
            primitives = list()
            for i in range(self.num_primitives):
                primitives.append(str(i))
            structs = enumerate_structures(max_depth)

            all_cat_trees = list()
            for s in structs:
                lts = generate_labeled_trees(s, operations, primitives)
                all_cat_trees.extend(lts)

        ix2cat = bidict.bidict()
        for t in all_cat_trees:
            ix2cat[len(ix2cat)] = t

        num_cats = len(ix2cat)

        # TODO change to np.inf?
        QUASI_INF = 10000000
        # rule_mask[:, 0:num_cats-1]: left child is functor
        # rule_mask[:, num_cats:2*num_cats-1]: right child is functor
        # rule_mask[:, 2*num_cats]: NULL (no category can be functor)
        # mask values are 0 if combination is possible, -QUASI_INF otherwise
        rule_mask = torch.full((num_cats, 2*num_cats+1), fill_value=-QUASI_INF)
        pc_null_ix = 2 * num_cats
        rule_mask[:, pc_null_ix] = 0
        can_be_root = torch.full((num_cats,), fill_value=-np.inf)

        # maps left-child functor category (index) to right-child argument
        # (value at index). If the left-child category on the left cannot be
        # a functor taking an argument on the right, the value will be lr_null_ix
        # (a dummy category)
        l2r = torch.empty(num_cats+1, dtype=torch.int64)
        lr_null_ix = num_cats
        l2r[lr_null_ix] = lr_null_ix
        # vice versa
        r2l = torch.empty(num_cats+1, dtype=torch.int64)
        r2l[lr_null_ix] = lr_null_ix
            
        for cat_ix in ix2cat:
            cat = ix2cat[cat_ix]
            cat_children = cat.children(cat.root)
            if len(cat_children) == 0:
                # only primitives cats can be the root of a parse tree
                can_be_root[cat_ix] = 0
                # primitives can't ever be functors
                l2r[cat_ix] = lr_null_ix
                r2l[cat_ix] = lr_null_ix
            else:
                assert len(cat_children) == 2
                # example: for the category {X-aY}-bZ, argument is Z
                # and result is X-aY
                result, arg = cat_children
                result = cat.subtree(result.identifier)
                result_ix = ix2cat.inverse[result]
                arg = cat.subtree(arg.identifier)
                arg_ix = ix2cat.inverse[arg]
                root_tag = cat.get_node(cat.root).tag 
                if root_tag == "-b":
                    # left child is functor
                    rule_mask[result_ix, cat_ix] = 0
                    rule_mask[result_ix, pc_null_ix] = -QUASI_INF
                    
                    l2r[cat_ix] = arg_ix
                    r2l[cat_ix] = lr_null_ix
                else:
                    assert root_tag == "-a"
                    # right child is functor
                    rule_mask[result_ix, cat_ix+num_cats] = 0
                    rule_mask[result_ix, pc_null_ix] = -QUASI_INF
                    r2l[cat_ix] = arg_ix
                    l2r[cat_ix] = lr_null_ix


        print("CEC num cats: {}".format(num_cats))
        self.num_cats = num_cats
        self.ix2cat = ix2cat
        print("CEC ix2cat: {}".format(ix2cat))
        self.l2r = l2r
        self.r2l = r2l
        self.rule_mask = rule_mask
        self.root_mask = can_be_root


    def forward(self, x, eval=False, argmax=False, use_mean=False, indices=None, set_grammar=True, return_ll=True, **kwargs):
        # x : batch x n
        if set_grammar:
            self.emission = None

            fake_emb = self.fake_emb
            num_cats = self.num_cats

            root_scores = F.log_softmax(
                self.root_mask+self.root_mlp(self.root_emb).squeeze(), dim=0
            )
            full_p0 = root_scores


            rule_scores = F.log_softmax(
                self.rule_mlp(fake_emb) + self.rule_mask, dim=1
            ) # C x (2C+1)

            # Note: column[2*num_cats] is for NULL,
            # which is thrown out here
            rule_score_l = rule_scores[:, :num_cats]
            rule_score_r = rule_scores[:, num_cats:2*num_cats]

            nt_emb = self.nt_emb
            # split_scores[:, 0] gives P(terminal=0 | cat)
            # split_scores[:, 1] gives P(terminal=1 | cat)
            split_scores = F.log_softmax(self.split_mlp(nt_emb), dim=1)

            full_G_l = rule_score_l + split_scores[:, 0][..., None]
            full_G_r = rule_score_r + split_scores[:, 0][..., None]

            self.parser.set_models(
                full_p0,
                full_G_l,
                full_G_r,
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
