import bidict, csv, math, numpy as np, torch, torch.nn.functional as F
from torch import nn
from cky_parser_sgd import BatchCKYParser
from char_coding_models import ResidualLayer, WordProbFCFixVocabCompound
from cg_type import CGNode, read_categories_from_file

QUASI_INF = 10000000.
DEBUG = True

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

        self.init_cooccurrences(config)
        printDebug("num words:", num_words)
        printDebug("num preds:", self.num_preds)
        self.word_emb = nn.Embedding(num_words, self.num_preds)
        printDebug("cooc:", self.cooc)

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

        #printDebug("ix2cat:", self.ix2cat)
        #printDebug("ix2cat_gen:", self.ix2cat_gen)

        self.parser = BatchCKYParser(
            ix2cat=self.ix2cat,
            ix2cat_gen=self.ix2cat_gen,
            lfunc_ixs=self.lfunc_ixs,
            rfunc_ixs=self.rfunc_ixs,
            qall=self.qall,
            qgen=self.qgen,
            device=self.device
        )

        self.sample_count = config.getint("sample_count")

    def init_cooccurrences(self, config):
        cooc = list()
        dir = config["cooccurrence_scores_dir"]
        for op in ["argument1", "argument2", "modifier"]:
            op_scores = list()
            for row in csv.reader(open(dir + '/' + op)):
                scores = [float(s) for s in row]
                op_scores.append(scores)
            cooc.append(op_scores)
        # shape after permuute: pred x pred x op
        cooc = torch.Tensor(cooc).permute((1, 2, 0))
        self.num_preds = cooc.shape[0]
        self.cooc = cooc


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
        genix2allix = list()
        for cat in self.gen_cats:
            allix = self.ix2cat.inv[cat]
            ix2cat_gen[len(ix2cat_gen)] = cat
            genix2allix.append(allix)
        self.ix2cat_gen = ix2cat_gen
        # dim: qgen
        self.genix2allix = torch.tensor(genix2allix).to(self.device)
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
                    lfunc_ix = 0
                rfunc = CGNode("-a", par, gen)
                if rfunc in self.ix2cat.inv:
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

        
    def forward(
            self, x, argmax=False, set_grammar=True,
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

            self.root_probs = root_probs
            self.parser.set_models(
                root_probs,
                full_G,
                split_probs=split_probs
            )

        # drop bos and eos
        words = x[:, 1:-1]
        x = self.emit_prob_model(x, self.nt_emb, set_grammar=set_grammar)

        if argmax:
            with torch.no_grad():
                logprob_list, vtree_list = \
                    self.parser.marginal(x, viterbi_flag=True)
            # TODO get pred score for viterbi tree
            return logprob_list, vtree_list
        else:
            # dim of logprob_list: batch_size
            logprob_list, _, left_chart, right_chart = self.parser.marginal(
                x, return_charts=True
            )
            #printDebug("sampling from chart next...")
            sampled_trees, tree_logprobs = self.sample_from_chart(left_chart, right_chart)
            #printDebug("sampled trees:", sampled_trees)
            #printDebug("sampled tree logprobs", tree_logprobs)
            # dim: batch_size x samples
            biased_logprobs = self.get_biased_logprobs(words, sampled_trees)
            #printDebug("sampled tree logprobs with bias:", biased_logprobs)
            # logprob_list is the marginal probabiity from the parse, and the
            # latter two terms are the (estimated) expected association score
            printDebug("forward chart logprobs:", logprob_list)
            printDebug("forward assoc score:", biased_logprobs.logsumexp(dim=1) - math.log(self.sample_count)) 
            return (logprob_list + biased_logprobs.logsumexp(dim=1) - math.log(self.sample_count)) * -1

    def sample_from_chart(self, left_chart, right_chart):
        printDebug("sampling...")
        # left and right chart dim: sentlen x sentlen x batch x qall
        sent_len = left_chart.shape[0]
        batch_size = left_chart.shape[2]
        # dim: qall x 4qgen
        # TODO maybe logsumexp later instead of exp now
        full_G = self.parser.full_G

        # gen_ixs[ix] gives the index for category
        # ix2cat_gen[ix] within ix2cat
        # dim: qgen
        gen_ixs = torch.full((self.qgen,), -1)
        for ix, cat in self.ix2cat_gen.items():
            allix = self.ix2cat.inv[cat]
            gen_ixs[ix] = allix

        all_sent_samples = list()
        all_sent_logliks = list()
        for sent in range(batch_size):
            sent_samples = list()
            sent_logliks = list()
            # dim: sentlen x sentlen x qall
            curr_lc = left_chart[..., sent, :]
            sent_len = curr_lc.shape[0]
            printDebug("\nsent index {}, len {}".format(sent, sent_len))
            # dim: sentlen x sentlen x qall
            curr_rc = right_chart[..., sent, :].flip(dims=[0])
            for sample in range(self.sample_count):
                printDebug("\tsample index:", sample)
                # queue containing (category, start, end) for constituents
                # that need to be split up further
                q = list()
                # list of split points that can be used to rebuild the tree
                # bottom-up for scoring
                splits = list()
                # sample root from bottom left cell of curr_lc
                # dim: qall
                root_dist = curr_lc[-1, 0] + self.root_probs
                # dim: qall
                root_dist = root_dist.exp()
                # dim: 1
                root = torch.multinomial(root_dist, num_samples=1)
                # differentiable alternative to multinomial
                #root = torch.nn.functional.gumbel_softmax(root_dist, hard=True)
                # dim: 1
                #root = root.argmax().unsqueeze(dim=0)
                #printDebug("root:", root)
                # dim: 1
                # clone is needed to not overwrite the left chart
                loglik = curr_lc[-1, 0, root.item()].clone()
                # dim: 1 x 4qgen
                curr_G = full_G[root]
                # dim: sentlen-1 x 4qgen
                curr_G = curr_G.expand(sent_len-1, -1)
                # prob of root cat expanding to gen cat via Aa operation
                # left child is gen cat; right child is implicit
                # dim: sentlen-1 x qgen
                curr_G_Aa = curr_G[:, :self.qgen]
                # dim: sentlen-1 x qgen
                curr_G_Ab = curr_G[:, self.qgen:2*self.qgen]
                # dim: sentlen-1 x qgen
                curr_G_Ma = curr_G[:, 2*self.qgen:3*self.qgen]
                # dim: sentlen-1 x qgen
                curr_G_Mb = curr_G[:, 3*self.qgen:]
                
                # dim: sent_len-1 x qgen
                curr_gen_ixs = gen_ixs.expand(sent_len-1, -1).to(self.device)

                # dim: sentlen-1 x qall
                lc_scores = curr_lc[:-1, 0]
                # dim: sentlen-1 x qall
                rc_scores = curr_rc[1:, -1]

                # dim: qgen
                # for generated right argument i, lfunc_ixs[i] is the index
                # of the functor category on the left, when root is the
                # result
                lfunc_ixs_curr = self.lfunc_ixs[root]
                # dim: sent_len-1 x qgen
                lfunc_ixs_curr = lfunc_ixs_curr.expand(sent_len-1, -1)

                # dim: sentlen-1 x qgen
                lc_scores_Aa = torch.gather(lc_scores, dim=1, index=curr_gen_ixs)
                # dim: sentlen-1 x qgen
                lc_scores_Ab = torch.gather(lc_scores, dim=1, index=lfunc_ixs_curr)
                # dim: sentlen-1 x qgen
                lc_scores_Ma = torch.gather(lc_scores, dim=1, index=curr_gen_ixs)
                # dim: sentlen-1 x 1
                # left child inherits parent's category
                lc_scores_Mb = lc_scores[:, root].expand(-1, self.qgen)

                # dim: qgen
                # for generated left argument i, rfunc_ixs[i] is the index
                # of the functor category on the right, when root is the
                # result
                rfunc_ixs_curr = self.rfunc_ixs[root]
                # dim: sent_len-1 x qgen
                rfunc_ixs_curr = rfunc_ixs_curr.expand(sent_len-1, -1)
                # dim: sent_len-1 x qgen
                rc_scores_Aa = torch.gather(rc_scores, dim=1, index=rfunc_ixs_curr)
                # dim: sent_len-1 x qgen
                rc_scores_Ab = torch.gather(rc_scores, dim=1, index=curr_gen_ixs)
                # dim: sentlen-1 x qgen
                # right child inherits parent's category
                rc_scores_Ma = rc_scores[:, root].expand(-1, self.qgen)
                # dim: sent_len-1 x qgen
                rc_scores_Mb = torch.gather(rc_scores, dim=1, index=curr_gen_ixs)

                # dim: sent_len-1 x qgen
                combined_scores_Aa = sum([curr_G_Aa, lc_scores_Aa, rc_scores_Aa])
                # dim: sent_len-1 x qgen
                combined_scores_Ab = sum([curr_G_Ab, lc_scores_Ab, rc_scores_Ab])
                # dim: sent_len-1 x qgen
                combined_scores_Ma = sum([curr_G_Ma, lc_scores_Ma, rc_scores_Ma])
                # dim: sent_len-1 x qgen
                combined_scores_Mb = sum([curr_G_Mb, lc_scores_Mb, rc_scores_Mb])
                # dim: 4 x sent_len-1 x qgen
                stacked_all = torch.stack([
                    combined_scores_Aa,
                    combined_scores_Ab,
                    combined_scores_Ma,
                    combined_scores_Mb
                ])
                # dim: sentlen-1
                combined_all = stacked_all.logsumexp(dim=0).logsumexp(dim=1)
                # dim: sentlen-1
                split_point_weights = combined_all.exp()

                # sample split point
                split_point = torch.multinomial(split_point_weights, num_samples=1)


                # dim: 4 x qgen
                # scores of all (qgen, op) options after split point is fixed
                scores_after_split = stacked_all[:, split_point, :].squeeze(dim=1)
                # dim: 4qgen
                scores_after_split = scores_after_split.reshape(4*self.qgen).exp()
                op_and_gencat = torch.multinomial(scores_after_split, num_samples=1)
                
                op = torch.div(op_and_gencat, self.qgen, rounding_mode="floor")
                #op = torch.div(op_and_gencat, self.qgen, rounding_mode="floor").item()
                gencat = op_and_gencat % self.qgen
                #gencat = (op_and_gencat % self.qgen).item()
                # change gencat to index within the set of all categories,
                # instead of iundex within the set of possible generated
                # categories
                gencat_allix = self.genix2allix[gencat]

                # dim: qgen
                rfunc_ixs_curr = self.rfunc_ixs[root].squeeze(dim=0)
                # dim: qgen
                lfunc_ixs_curr = self.lfunc_ixs[root].squeeze(dim=0)
                # dim: qgen
                root_repeated = root.repeat(self.qgen)
                # dim: 4 x qgen
                all_impcats = torch.stack([
                    rfunc_ixs_curr,
                    lfunc_ixs_curr,
                    root_repeated,
                    root_repeated
                ])
                impcat = all_impcats[op, gencat]
                # Aa or Ma
                if op % 2 == 0:
                    lcat = gencat_allix
                    rcat = impcat
                # Ab or Mb
                else:
                    lcat = impcat
                    rcat = gencat_allix

                # for Aa or Ab, need to keep track of how deep the implicit
                # functor category is. This determines which cooccurrence 
                # matrix is used
                if op < 2:
                    functor_depth = self.ix2cat[impcat.item()].get_depth()
                else:
                    functor_depth = -1
                
                lcat_start = 0
                lcat_end = split_point.item() + 1
                rcat_start = split_point.item() + 1
                rcat_end = sent_len

                #printDebug("gencat:", self.ix2cat[gencat_allix.item()])
                #printDebug("implicit cat:", self.ix2cat[impcat.item()])

                printDebug("root cat:", self.ix2cat[root.item()])
                printDebug("lcat: {}; span: {} - {}".format(self.ix2cat[lcat.item()], lcat_start, lcat_end))
                printDebug("rcat: {}; span: {} - {}".format(self.ix2cat[rcat.item()], rcat_start, rcat_end))
                printDebug("operation:", op.item())
                splits.append((op.item(), functor_depth, split_point.item() + 1))

                #printDebug("adding to q: {} - {}".format(lcat_start, lcat_end))
                #printDebug("adding to q: {} - {}".format(rcat_start, rcat_end))
                q.append((lcat, lcat_start, lcat_end))
                q.append((rcat, rcat_start, rcat_end))
                # then sample left child cat, right child cat
                # - add (cat, start, end) to q for L and R children
                while len(q) > 0:
                    #printDebug("q length before pop:", len(q))
                    curr_cat, curr_start, curr_end = q.pop(0)
                    #printDebug("curr constituent range: {} - {}".format(curr_start, curr_end))
                    # leaf node: no further branching needed
                    if curr_start + 1 == curr_end:
                        #printDebug("word reached, not splitting further")
                        continue
                    ijdiff = curr_end - curr_start

                    # TODO update dims to ijdiff-1 instead of sent_len-1
                    # dim: 1 x 4qgen
                    curr_G = full_G[curr_cat]
                    # dim: sentlen-1 x 4qgen
                    curr_G = curr_G.expand(ijdiff-1, -1)
                    # prob of root cat expanding to gen cat via Aa operation
                    # left child is gen cat; right child is implicit
                    # dim: sentlen-1 x qgen
                    curr_G_Aa = curr_G[:, :self.qgen]
                    # dim: sentlen-1 x qgen
                    curr_G_Ab = curr_G[:, self.qgen:2*self.qgen]
                    # dim: sentlen-1 x qgen
                    curr_G_Ma = curr_G[:, 2*self.qgen:3*self.qgen]
                    # dim: sentlen-1 x qgen
                    curr_G_Mb = curr_G[:, 3*self.qgen:]

                    # dim: ijdiff-1 x qgen
                    curr_gen_ixs = gen_ixs.expand(ijdiff-1, -1).to(self.device)

                    # dim: ijdiff-1 x qgen
                    lc_scores = curr_lc[:ijdiff-1, curr_start]
                    # dim: ijdiff-1 x qgen
                    rc_scores = curr_rc[-(ijdiff-1):, curr_end-1]
                    # dim: ijdiff-1 x qgen
                    lfunc_ixs_curr = self.lfunc_ixs[curr_cat].expand(ijdiff-1, -1)

                    # dim: sentlen-1 x qgen
                    lc_scores_Aa = torch.gather(lc_scores, dim=1, index=curr_gen_ixs)
                    # dim: sentlen-1 x qgen
                    lc_scores_Ab = torch.gather(lc_scores, dim=1, index=lfunc_ixs_curr)
                    # dim: sentlen-1 x qgen
                    lc_scores_Ma = torch.gather(lc_scores, dim=1, index=curr_gen_ixs)
                    # dim: sentlen-1 x 1
                    # left child inherits parent's category
                    lc_scores_Mb = lc_scores[:, curr_cat].expand(-1, self.qgen)

                    # dim: qgen
                    rfunc_ixs_curr = self.rfunc_ixs[curr_cat]
                    # dim: sent_len-1 x qgen
                    rfunc_ixs_curr = rfunc_ixs_curr.expand(ijdiff-1, -1)
                    # dim: sent_len-1 x qgen
                    rc_scores_Aa = torch.gather(rc_scores, dim=1, index=rfunc_ixs_curr)
                    # dim: sent_len-1 x qgen
                    rc_scores_Ab = torch.gather(rc_scores, dim=1, index=curr_gen_ixs)
                    # dim: sentlen-1 x qgen
                    # right child inherits parent's category
                    rc_scores_Ma = rc_scores[:, curr_cat].expand(-1, self.qgen)
                    # dim: sent_len-1 x qgen
                    rc_scores_Mb = torch.gather(rc_scores, dim=1, index=curr_gen_ixs)

                    # dim: sent_len-1 x qgen
                    combined_scores_Aa = sum([curr_G_Aa, lc_scores_Aa, rc_scores_Aa])
                    # dim: sent_len-1 x qgen
                    combined_scores_Ab = sum([curr_G_Ab, lc_scores_Ab, rc_scores_Ab])
                    # dim: sent_len-1 x qgen
                    combined_scores_Ma = sum([curr_G_Ma, lc_scores_Ma, rc_scores_Ma])
                    # dim: sent_len-1 x qgen
                    combined_scores_Mb = sum([curr_G_Mb, lc_scores_Mb, rc_scores_Mb])
                    # dim: 4 x sent_len-1 x qgen
                    stacked_all = torch.stack([
                        combined_scores_Aa,
                        combined_scores_Ab,
                        combined_scores_Ma,
                        combined_scores_Mb
                    ])
                    # dim: sentlen-1
                    combined_all = stacked_all.logsumexp(dim=0).logsumexp(dim=1)
                    # dim: sentlen-1
                    split_point_weights = combined_all.exp()

                    # sample split point
                    split_point = torch.multinomial(split_point_weights, num_samples=1)

                    # dim: 4 x qgen
                    # scores of all (qgen, op) options after split point is fixed
                    scores_after_split = stacked_all[:, split_point, :].squeeze(dim=1)
                    # dim: 4qgen
                    scores_after_split = scores_after_split.reshape(4*self.qgen)
                    op_and_gencat = torch.multinomial(scores_after_split.exp(), num_samples=1)
                    curr_loglik = scores_after_split[op_and_gencat.item()]
                    loglik += curr_loglik
                    
                    op = torch.div(op_and_gencat, self.qgen, rounding_mode="floor")
                    gencat = op_and_gencat % self.qgen
                    gencat_allix = self.genix2allix[gencat]
                    #printDebug("gencat_allix:", gencat_allix)

                    # dim: qgen
                    rfunc_ixs_curr = self.rfunc_ixs[curr_cat].squeeze(dim=0)
                    # dim: qgen
                    lfunc_ixs_curr = self.lfunc_ixs[curr_cat].squeeze(dim=0)
                    # dim: qgen
                    curr_cat_repeated = curr_cat.repeat(self.qgen)
                    # dim: 4 x qgen
                    all_impcats = torch.stack([
                        rfunc_ixs_curr,
                        lfunc_ixs_curr,
                        curr_cat_repeated,
                        curr_cat_repeated
                    ])
                    impcat = all_impcats[op, gencat]
                    # Aa or Ma
                    if op % 2 == 0:
                        lcat = gencat_allix
                        rcat = impcat
                    else:
                        lcat = impcat
                        rcat = gencat_allix

                    if op < 2:
                        printDebug("functor cat for arg attachment:", self.ix2cat[impcat.item()])
                        printDebug("depth:", self.ix2cat[impcat.item()].get_depth())
                        functor_depth = self.ix2cat[impcat.item()].get_depth()
                    else:
                        functor_depth = -1

                    lcat_start = curr_start
                    lcat_end = curr_start + split_point.item() + 1
                    rcat_start = curr_start + split_point.item() + 1 
                    rcat_end = curr_end

                    printDebug("curr cat:", self.ix2cat[curr_cat.item()])
                    printDebug("lcat: {}; span: {} - {}".format(self.ix2cat[lcat.item()], lcat_start, lcat_end))
                    printDebug("rcat: {}; span: {} - {}".format(self.ix2cat[rcat.item()], rcat_start, rcat_end))
                    printDebug("operation:", op.item())

                    # lcat_start + split_point.item() + 1 is the location of the
                    # split point within the entire sentence (instead of within
                    # the current constituent being split)
                    splits.append(
                        (op.item(), functor_depth, lcat_start+split_point.item()+1)
                    )

                    q.append((lcat, lcat_start, lcat_end))
                    q.append((rcat, rcat_start, rcat_end))
                sent_samples.append(splits)
                sent_logliks.append(loglik)
            all_sent_samples.append(sent_samples)
            all_sent_logliks.append(sent_logliks)
        # dim: batch x samples
        all_sent_logliks = torch.tensor(all_sent_logliks)
        return all_sent_samples, all_sent_logliks


    def compose_vectors(
            self, op, arg, split, curr_vecs, constituent_boundaries, score
        ):
        # dim of curr_vecs: sentlen x d
        # dim of constituent_boundaries: sentlen x 2
        # the new constituent span is (lconst_lower, rconst_upper)
        lconst_lower, _ = constituent_boundaries[split-1]
        _, rconst_upper = constituent_boundaries[split]
        constituent_boundaries[lconst_lower][0] = lconst_lower
        constituent_boundaries[lconst_lower][1] = rconst_upper
        constituent_boundaries[rconst_upper-1][0] = lconst_lower
        constituent_boundaries[rconst_upper-1][1] = rconst_upper
        # Aa - preceding argument attachment
        if op == 0:
            # dim: d
            argument = curr_vecs[split-1].clone() #.softmax(dim=0)
            # dim: d
            functor = curr_vecs[split].clone() #.softmax(dim=0)
            if arg == 1:
                # dim: d x d
                cooc = self.cooc[..., 0].clone()
            else:
                assert arg == 2
                # dim: d x d
                cooc = self.cooc[..., 1]
            prod = torch.matmul(cooc, argument)
            new_score = torch.matmul(functor, prod)
            #new_score = functor.matmul(cooc.matmul(argument))
            # functor's predicate becomes the predicate for the whole
            # constituent
            # update predicates at the constituent boundaries
            curr_vecs[lconst_lower] = functor
            curr_vecs[rconst_upper-1] = functor
        # Ab - succeeding argument attachment
        elif op == 1:
            # dim: d
            functor = curr_vecs[split-1].clone() #.softmax(dim=0)
            # dim: d
            argument = curr_vecs[split].clone() #.softmax(dim=0)
            if arg == 1:
                # dim: d x d
                cooc = self.cooc[..., 0].clone()
            else:
                assert arg == 2
                # dim: d x d
                cooc = self.cooc[..., 1]
            prod = torch.matmul(cooc, argument)
            new_score = torch.matmul(functor, prod)
            #new_score = functor.matmul(cooc.matmul(argument))
            #new_score = functor.matmul(cooc.matmul(argument))
            curr_vecs[lconst_lower] = functor
            curr_vecs[rconst_upper-1] = functor
        # Ma - preceding modifier attachment
        elif op == 2:
            assert arg == -1
            # dim: d
            modifier = curr_vecs[split-1].clone() #.softmax(dim=0)
            # dim: d
            modificand = curr_vecs[split].clone() #.softmax(dim=0)
            # dim: d x d
            cooc = self.cooc[..., 2].clone()
            #prod = cooc.matmul(modificand)
            #printDebug("cooc shape:", cooc.shape)
            #printDebug("modificand shape:", modificand.shape)
            prod = torch.matmul(cooc, modificand)
            #new_score = modifier.matmul(prod)
            new_score = torch.matmul(modifier, prod)
            #prod = cooc.matmul(modificand)
            #new_score = modifier.matmul(prod)
            #new_score = modifier.matmul(cooc.matmul(modificand))
            curr_vecs[lconst_lower] = modificand
            curr_vecs[rconst_upper-1] = modificand
        # Mb - succeeding modifier attachment
        elif op == 3:
            assert arg == -1
            # dim: d
            modificand = curr_vecs[split-1].clone() #.softmax(dim=0)
            # dim: d
            modifier = curr_vecs[split].clone() #.softmax(dim=0)
            # dim: d x d
            cooc = self.cooc[..., 2].clone()
            #prod = cooc.matmul(modificand)
            #printDebug("cooc shape:", cooc.shape)
            #printDebug("modificand shape:", modificand.shape)
            prod = torch.matmul(cooc, modificand)
            #new_score = modifier.matmul(prod)
            new_score = torch.matmul(modifier, prod)
            #new_score = modifier.matmul(cooc.matmul(modificand))
            curr_vecs[lconst_lower] = modificand
            curr_vecs[rconst_upper-1] = modificand
        #printDebug("new score:", new_score)
        score += new_score
    
    def get_biased_logprobs(self, words, sampled_trees):
        printDebug("getting biased logprobs...")
        # elements of sampled trees: batch items
        # elements of sampled_trees[i]: samples for a single batch item
        # elements of sampled_trees[i][j]: split decisions for a single
        # sampled tree
        #printDebug("words shape:", words.shape)
        # dim of words: batch x sentlen
        #printDebug("words:", words)
        # dim: batch_size x samples
        batchsize = len(sampled_trees)
        samples = len(sampled_trees[0])
        biased_logprobs = torch.zeros((batchsize, samples))
        #printDebug("biased_logprobs shape:", biased_logprobs.shape)
        #batchsize = words.shape[0]
        sentlen = words.shape[1]
        # dim: batch x sentlen x d
        all_word_embs = self.word_emb(words)
        # NOTE: for reasons I don't understand, it is not possible to
        # softmax all_word_embs in a single line, e.g.
        # all_word_embs = self.word_emb(words).softmax(dim=0).
        # Instead we softmax it row by row.
        # Softmaxing means that each word vector can be interpreted as a
        # distribution over predicates
        for i in range(sentlen):
            # dim: batch x d
            curr_word_embs = all_word_embs[:, i].clone().softmax(dim=1)
            all_word_embs[:, i] = curr_word_embs
        printDebug("all_word_embs:", all_word_embs)
        for i, sent in enumerate(sampled_trees):
            printDebug("sampled tree ix:", i)
            # dim: sentlen x d
            word_embs = all_word_embs[i]
            #printDebug("embeddings:", word_embs)
            for j, sample in enumerate(sent):
                printDebug("sample ix:", j)
                # dim: sent_len x d
                curr_vecs = word_embs.clone()
                constituent_boundaries = list([i, i+1] for i in range(sentlen))
                # dim: sent_len x 2
                constituent_boundaries = torch.tensor(constituent_boundaries)
                printDebug("current vecs:", curr_vecs)
                printDebug("constituent_boundaries:", constituent_boundaries)
                # reverse the steps to allow building the tree bottom-up
                # instead of splitting top-down
                steps = reversed(sample)
                score = torch.Tensor([0])
                for op, arg, split in steps:
                    printDebug("current op, arg, split:", op, arg, split)
                    self.compose_vectors(
                        op, arg, split, curr_vecs, constituent_boundaries, score
                    )
                    printDebug("current vecs now:", curr_vecs)
                    printDebug("constituent boundaries now:", constituent_boundaries)
                printDebug("score after steps:", score)
                biased_logprobs[i, j] = score[0]
        return biased_logprobs
