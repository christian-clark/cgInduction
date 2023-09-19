import torch, torch.nn as nn, numpy as np
from torch.nn import functional as F

DEBUG = True

def printDebug(*args, **kwargs):
    if DEBUG:
        print("DEBUG: ", end="")
        print(*args, **kwargs)


# TODO make bias configurable
def get_fixed_noun_category_mask(
        pos_indices, num_cats, primitive_cats, device, bias=1000.
    ):
    """Creates a masking matrix that biases the model to assign a
    primitive category to nouns.

    pos_indices is a matrix of dimension num_sents x num_words, where
    pos_indices[i, j] is a 0 or 1 indicating whether word j of sentence
    i is a noun."""

    num_sents = pos_indices.shape[0]
    num_words = pos_indices.shape[1]
    mask_vals = torch.Tensor([0, -bias]).to(device)
    mask_vals = mask_vals.tile(num_sents, 1)
    # this part of the mask blocks out non-noun categories for noun words
    # noun category is cat with index 0
    # sent x words x num_cats-1
    #mask_non_noun_cat = torch.gather(mask_vals, 1, pos_indices).unsqueeze(dim=-1).expand(-1, -1, num_cats-1)
    # and this part of the mask is for not blocking the noun category
    #mask_noun_cat = torch.zeros([num_sents, num_words, 1]).to(device)
    #mask = torch.cat((mask_noun_cat, mask_non_noun_cat), dim=2)

    # TODO move this into class so it isn't rebuilt over and over
    category_nomask_vector = torch.zeros((num_cats,)).to(device)
    category_mask_vector = torch.full((num_cats,), -bias).to(device)
    for cat in primitive_cats:
        category_mask_vector[cat] = 0
    category_nomask_mask = \
        torch.stack([category_nomask_vector, category_mask_vector], dim=1)
    # sents x words x cats x 2
    category_nomask_mask = category_nomask_mask.reshape(1, 1, num_cats, 2) \
                                               .expand(num_sents, num_words, -1, -1)

    # sents x words x cats x 1
    pos_indices_expanded = pos_indices.unsqueeze(dim=-1) \
                                      .expand(-1, -1, num_cats) \
                                      .unsqueeze(dim=-1)

    # sents x words x cats
    mask = torch.gather(category_nomask_mask, 3, pos_indices_expanded).squeeze(dim=3)
    return mask

#    # TODO can almost certainly be made more efficient using gather
#    per_category_masks = list()
#    for i in range(num_cats):
#        if i in primitive_cats:
#            mask_val = 0.
#        else:
#            mask_val = -bias
#        vals = torch.Tensor([0, mask_val]).to(device)
#        vals = vals.tile(num_sents, 1)
#        per_category_mask = torch.gather(vals, 1, pos_indices).unsqueeze(dim=-1)
#        per_category_masks.append(per_category_mask)
#
#    return torch.cat(per_category_masks, dim=2)


class ResidualLayer(nn.Module): # from kim
    def __init__(self, in_dim=100,
                 out_dim=100):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return F.relu(self.lin2(F.relu(self.lin1(x)))) + x


class CharProbRNN(nn.Module):
    def __init__(
            self,
            num_chars,
            primitive_cats,
            state_dim=256,
            hidden_size=256,
            num_layers=4,
            dropout=0.,
            device="cpu"
        ):
        super(CharProbRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.primitive_cats = primitive_cats
        self.device = device

        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        # self.rnn = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

        self.top_fc = nn.Linear(hidden_size, num_chars)
        self.char_embs = nn.Embedding(num_chars, hidden_size)

        # self.cat_emb_expansion = nn.Sequential(nn.Linear(state_dim, hidden_size), nn.ReLU())
        self.cat_emb_expansion = nn.Sequential(nn.Linear(state_dim, hidden_size*num_layers), nn.ReLU())

        torch.nn.init.kaiming_normal_(self.char_embs.weight.data)

    def forward(self, chars, pos, cat_embs, set_grammar=True): # does not use set pcfg
        # TODO noun mask here
        char_embs, cat_embs = self.prep_input(chars, cat_embs)
        Hs = []
        lens = 0
        for cat_tensor in cat_embs: # each cat at one time
            # for simple RNNs
            # # cat_tensor is batch, dim
            # cat_tensor = cat_tensor.unsqueeze(0).expand(self.num_layers, -1, -1)
            # cat_tensor = self.cat_emb_expansion(cat_tensor)
            # all_hs, _ = self.rnn.forward(char_embs, cat_tensor)
            # all_hs = nn.utils.rnn.pad_packed_sequence(all_hs) # len, batch, embs
            # Hs.append(all_hs[0].transpose(0,1))
            # lens = all_hs[1]

            # for LSTMs with 3d linears
            cat_tensor = self.cat_emb_expansion(cat_tensor) # batch, hidden*numlayers
            cat_tensor = cat_tensor.reshape(cat_tensor.shape[0], self.hidden_size, -1)
            cat_tensor = cat_tensor.permute(2, 0, 1)
            h0_tensor = torch.zeros_like(cat_tensor)
            all_hs, _ = self.rnn.forward(char_embs, (h0_tensor, cat_tensor))
            all_hs = nn.utils.rnn.pad_packed_sequence(all_hs) # len, batch, embs
            Hs.append(all_hs[0].transpose(0,1))
            lens = all_hs[1]

        Hses = torch.stack(Hs, 0)
        # Hses = nn.functional.relu(Hses)
        scores = self.top_fc.forward(Hses) # cats, batch, num_chars_in_word, num_chars
        logprobs = torch.nn.functional.log_softmax(scores, dim=-1)
        total_logprobs = []

        for idx, length in enumerate(lens.tolist()):
            this_word_logprobs = logprobs[:, idx, :length, :] # cats, (batch_scalar), num_chars_in_word, num_chars
            sent_id = idx // len(chars[0])
            word_id = idx % len(chars[0])
            targets = chars[sent_id][word_id][1:]
            this_word_logprobs = this_word_logprobs[:, range(this_word_logprobs.shape[1]), targets]  # cats, num_chars_in_word
            total_logprobs.append(this_word_logprobs.sum(-1)) # cats
        total_logprobs = torch.stack(total_logprobs, dim=0) # batch, cats
        total_logprobs = total_logprobs.reshape(len(chars), -1, total_logprobs.shape[1]) # sentbatch, wordbatch, cats
        num_cats = total_logprobs.shape[2]
        pos_indices = pos[:, 1:-1]
        mask = get_fixed_noun_category_mask(
            pos_indices, num_cats, self.primitive_cats, self.device
        )

        masked_logprobs = total_logprobs + mask

        #return total_logprobs
        return masked_logprobs

    def prep_input(self, chars, cat_embs):
        # cat_embs is num_cat, cat_dim
        # chars is num_words, word/char_tensor
        embeddings = []
        for i in range(len(chars)):
            for j in range(len(chars[i])):
                embeddings.append(self.char_embs.forward(chars[i][j][:-1])) # no word end token
        packed_char_embs = nn.utils.rnn.pack_sequence(embeddings, enforce_sorted=False) # len, batch, embs
        expanded_cat_embs = cat_embs.unsqueeze(1).expand(-1, packed_char_embs.data.size(0), -1) # numcat,batch, catdim

        return packed_char_embs, expanded_cat_embs


class WordProbFCFixVocabCompound(nn.Module):
    def __init__(
            self, num_words, state_dim, primitive_cats, dropout=0.0, device="cpu"
        ):
        super(WordProbFCFixVocabCompound, self).__init__()
        self.primitive_cats = primitive_cats
        self.device = device
        self.fc = nn.Sequential(nn.Linear(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       nn.Linear(state_dim, num_words))

    def forward(self, words, pos, cat_embs, set_grammar=True):
        if set_grammar:
            dist = nn.functional.log_softmax(self.fc(cat_embs), 1).t() # vocab x cats
            self.dist = dist
        else:
            pass

        word_indices = words[:, 1:-1]
        logprobs = self.dist[word_indices, :] # sent, word, cats; get rid of bos and eos

        pos_indices = pos[:, 1:-1]
        num_cats = self.dist.shape[1]
        mask = get_fixed_noun_category_mask(
            pos_indices, num_cats, self.primitive_cats, self.device
        )

        masked_logprobs = logprobs + mask
        #return logprobs
        return masked_logprobs

