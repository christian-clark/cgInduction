import torch.nn as nn
from torch.nn import functional as F


class ResidualLayer(nn.Module): # from kim
    def __init__(self, in_dim=100,
                 out_dim=100):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return F.relu(self.lin2(F.relu(self.lin1(x)))) + x


class WordProbFCFixVocabCompound(nn.Module):
    def __init__(self, num_words, state_dim):
        super(WordProbFCFixVocabCompound, self).__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       nn.Linear(state_dim, num_words))

    def forward(self, words, cat_embs, set_grammar=True):
        if set_grammar:
            dist = nn.functional.log_softmax(self.fc(cat_embs), 1).t() # vocab, cats
            self.dist = dist
        else:
            pass
        word_indices = words[:, 1:-1]

        logprobs = self.dist[word_indices, :] # sent, word, cats; get rid of bos and eos
        return logprobs

