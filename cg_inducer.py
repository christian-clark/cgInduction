import bidict, numpy as np, torch, torch.nn.functional as F
from torch import nn
from cky_parser_sgd import BatchCKYParser
from char_coding_models import ResidualLayer, WordProbFCFixVocabCompound
from cg_type import CGNode, read_categories_from_file

QUASI_INF = 10000000.
DEBUG = False

def printDebug(*args, **kwargs):
    if DEBUG:
        print("DEBUG: ", end="")
        print(*args, **kwargs)


class BasicCGInducer(nn.Module):
    def __init__(self, config, num_words):
        super(BasicCGInducer, self).__init__()
        self.state_dim = config.getint("state_dim")
        self.model_type = config["model_type"]
        self.device = config["device"]
        self.eval_device = config["eval_device"]

        # note: this model only supports a word-level model
        # jin et al define a character model that performs better but is
        # slower
        self.emit_prob_model = WordProbFCFixVocabCompound(
            num_words, self.state_dim
        )
        # specify a file with a list of categories
        # (used in 2024 COLING paper)
        self.cats_list = config.get("category_list")
        self.init_cats()
        self.init_masks()

        # "embeddings" for the categories are just one-hot vectors
        # these are used for result categories
        self.fake_emb = nn.Parameter(torch.eye(self.qall))
        state_dim = self.state_dim
        # actual embeddings are used to calculate split scores
        # (i.e. prob of terminal vs nonterminal)
        # dim: qall x D
        self.nt_emb = nn.Parameter(
            torch.randn(self.qall, state_dim)
        )
        # maps parent cat to arg_cat x {Aa, Ab, Ma, Mb}
        self.rule_mlp = nn.Linear(self.qall, 4*self.qgen)
        self.root_emb = nn.Parameter(torch.eye(1)).to(self.device)
        self.root_mlp = nn.Linear(1, self.qall).to(self.device)

        # decides terminal or nonterminal
        self.split_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            ResidualLayer(state_dim, state_dim),
            nn.Linear(state_dim, 2)
        ).to(self.device)

        self.parser = BatchCKYParser(
            ix2cat=self.ix2cat,
            ix2cat_gen=self.ix2cat_gen,
            lfunc_ixs=self.lfunc_ixs,
            rfunc_ixs=self.rfunc_ixs,
            #TODO needed?
            #larg_mask=self.larg_mask,
            #rarg_mask=self.rarg_mask,
            qall=self.qall,
            qgen=self.qgen,
            device=self.device
        )


    def init_cats(self):
        # arg_cats is needed in init_masks
        # TODO is res_cats needed?
        all_cats, gen_cats, arg_cats, _ = read_categories_from_file(
            self.cats_list
        )

        self.all_cats = sorted(all_cats)
        self.qall = len(self.all_cats)
        ix2cat = bidict.bidict()
        for cat in self.all_cats:
            ix2cat[len(ix2cat)] = cat
        self.ix2cat = ix2cat
        self.gen_cats = sorted(gen_cats)
        self.qgen = len(self.gen_cats)
        ix2cat_gen = bidict.bidict()
        for cat in self.gen_cats:
            ix2cat_gen[len(ix2cat_gen)] = cat
        self.ix2cat_gen = ix2cat_gen
        self.arg_cats = arg_cats

        # given an result cat index (i) and an argument cat 
        # index (j), lfunc_ixs[i, j] gives the functor cat that
        # takes cat j as a right argument and returns cat i
        # e.g. for the rule V -> V-bN N:
        # lfunc_ixs[V, N] = V-bN
        # NOTE: this doesn't cover modifiers. Given a result cat and a
        # modifier cat, the modifcand cat will be the same as the result cat
        lfunc_ixs = torch.zeros(
            self.qall, self.qgen, dtype=torch.int64
        )

        # same idea but functor appears on the right
        # e.g. for the rule V -> N V-aN:
        # rfunc_ixs[V, N] = V-aN
        rfunc_ixs = torch.zeros(
            self.qall, self.qgen, dtype=torch.int64
        )

        for par_ix, par in self.ix2cat.items():
            for gen_ix, gen in self.ix2cat_gen.items():
                lfunc = CGNode("-b", par, gen)
                if lfunc in self.ix2cat.inv:
                    lfunc_ix = self.ix2cat.inv[lfunc]
                else:
                    # TODO needed?
                    #rarg_mask[res_ix, arg_ix] = -QUASI_INF
                    lfunc_ix = 0
                rfunc = CGNode("-a", par, gen)
                if rfunc in self.ix2cat.inv:
                    # TODO needed?
                    #larg_mask[res_ix, arg_ix] = -QUASI_INF
                    rfunc_ix = self.ix2cat.inv[rfunc]
                else:
                    rfunc_ix = 0
                lfunc_ixs[par_ix, gen_ix] = lfunc_ix
                rfunc_ixs[par_ix, gen_ix] = rfunc_ix

        self.ix2cat = ix2cat
        self.ix2cat_gen = ix2cat_gen
        self.lfunc_ixs = lfunc_ixs.to(self.device)
        self.rfunc_ixs = rfunc_ixs.to(self.device)


    def init_masks(self):
        # lgen_mask and rgen_mask block impossible parent-gen pairs
        # with the generated child on the left and right respectively
        # e.g. if 0 is in par_cats and 1 is in gen_cats but 0/1 is not in
        # all cats, then (0, 1) should be blocked as a parent-gen pair
        # these only make a difference if categories are read in from
        # a file, not if they're generated by depth
        # dim: qall x qgen
        lfunc_mask = torch.zeros(
            self.qall, self.qgen, dtype=torch.float32
        ).to(self.device)
        # dim: qall x qgen
        rfunc_mask = torch.zeros(
            self.qall, self.qgen, dtype=torch.float32
        ).to(self.device)

        for par_ix, par in self.ix2cat.items():
            for gen_ix, gen in self.ix2cat_gen.items():
                if gen not in self.arg_cats:
                    lfunc_mask[par_ix, gen_ix] = -QUASI_INF
                    rfunc_mask[par_ix, gen_ix] = -QUASI_INF
                else:
                    # TODO don't hard-code operator
                    lfunc = CGNode("-b", par, gen)
                    if lfunc not in self.all_cats:
                        lfunc_mask[par_ix, gen_ix] = -QUASI_INF
                    rfunc = CGNode("-a", par, gen)
                    if rfunc not in self.all_cats:
                        rfunc_mask[par_ix, gen_ix] = -QUASI_INF

        # blocks categories that aren't in form u-av from being
        # used as modifiers
        # dim: qgen
        mod_mask = torch.zeros(
            self.qgen, dtype=torch.float32
        ).to(self.device)

        for gen_ix, gen in self.ix2cat_gen.items():
            if not gen.is_modifier():
                mod_mask[gen_ix] = -QUASI_INF

        # only allow primitive categories to be at the root of the parse
        # tree
        # dim: qall
        root_mask = torch.full((self.qall,), fill_value=-np.inf).to(self.device)
        for cat_ix, cat in self.ix2cat.items():
            if cat.is_primitive():
                root_mask[cat_ix] = 0

        self.lfunc_mask = lfunc_mask
        self.rfunc_mask = rfunc_mask
        self.mod_mask = mod_mask
        self.root_mask = root_mask

        
    # TODO clean up arguments
    def forward(
            self, x, eval=False, argmax=False, indices=None,
            set_grammar=True, return_ll=True
        ):
        # x : batch x n
        if set_grammar:
            # to assign equal probability to all possible root categories
            #root_probs = F.log_softmax(self.root_mask, dim=0)
            # dim: qall
            root_probs = F.log_softmax(
                self.root_mask+self.root_mlp(self.root_emb).squeeze(), dim=0
            )

            # dim: qall x 4qgen
            rule_scores = self.rule_mlp(self.fake_emb)
            # mask impossible pairs for Aa operation
            rule_scores[:, :self.qgen] += self.rfunc_mask
            # mask impossible pairs for Ab operation
            rule_scores[:, self.qgen:2*self.qgen] += self.lfunc_mask
            # mask impossible pairs for Ma operation
            rule_scores[:, 2*self.qgen:3*self.qgen] += self.mod_mask[None, :]
            # mask impossible pairs for Mb operation
            rule_scores[:, 3*self.qgen:] += self.mod_mask[None, :]
            # dim: qres x 4qarg
            rule_probs = F.log_softmax(rule_scores, dim=1)

            # split_probs[:, 0] gives P(terminal=0 | cat)
            # split_probs[:, 1] gives P(terminal=1 | cat)
            # dim: qall x 2
            split_probs = F.log_softmax(self.split_mlp(self.nt_emb), dim=1)
            # dim: qall
            nont_probs = split_probs[:, 0]

            # dim: qall x 4qgen
            full_G = rule_probs + nont_probs[..., None]

            self.parser.set_models(
                root_probs,
                full_G,
                split_probs=split_probs
            )

        x = self.emit_prob_model(x, self.nt_emb, set_grammar=set_grammar)

        if argmax:
            printDebug("inducer_x")
            printDebug(x.flatten()[:20])
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
