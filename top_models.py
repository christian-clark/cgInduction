import logging
import torch.nn as nn
import torch.distributions
from torch.utils.tensorboard import SummaryWriter

DEBUG = False

def printDebug(*args, **kwargs):
    if DEBUG:
        print("DEBUG: ", end="")
        print(*args, **kwargs)


class TopModel(nn.Module):
    def __init__(self, inducer, writer:SummaryWriter=None):
        super(TopModel, self).__init__()
        self.inducer = inducer
        self.writer = writer


    def forward(self, word_inp, chars_var_inp):
        if self.inducer.model_type == "char":
            logprob_list, _, _, _ = self.inducer.forward(
                chars_var_inp, words=word_inp
            )
        else:
            assert self.inducer.model_type == 'word'
            logprob_list, _, _, _ = self.inducer.forward(word_inp)
        structure_loss = torch.sum(logprob_list, dim=0)
        return structure_loss


    def get_category_entropy_loss(self):
        # shape: vocab x predcats
        # NOTE super hacky, but the 6: removes indices 0 through 5 which
        # are special tokens (oov, bos, etc.)
        logdist = self.inducer.emit_prob_model.dist[6:]
        printDebug("logdist:", logdist)
        per_word_total = logdist.logsumexp(dim=1, keepdim=True)
        normalized_logdist = logdist - per_word_total
        printDebug("normalized_logdist:", normalized_logdist)
        normalized_dist = torch.exp(normalized_logdist)
        printDebug("normalized_dist:", normalized_dist)
        # sum_x p(x) log p(x)
        entropy = torch.sum(normalized_dist*normalized_logdist, dim=1)
        printDebug("entropy:", entropy)
        mean_entropy = torch.mean(entropy)
        printDebug("mean_entropy:", mean_entropy)
        return -1 * mean_entropy


    def parse(self, word_inp, chars_var_inp, indices, eval=False, set_grammar=True):
        if self.inducer.model_type == 'char':
            structure_loss, vtree_list, _, _ = self.inducer.forward(
                chars_var_inp,
                eval,
                argmax=True,
                indices=indices,
                set_grammar=set_grammar
            )
        else:
            assert self.inducer.model_type == 'word'
            printDebug("word input: {}".format(word_inp))
            structure_loss, vtree_list, _, _ = self.inducer.forward(
                word_inp,
                eval,
                argmax=True,
                indices=indices,
                set_grammar=set_grammar
            )
        return structure_loss.sum().item(), vtree_list


    def likelihood(self, word_inp, chars_var_inp, indices, set_grammar=True):
        if self.inducer.model_type == 'char':
            structure_loss = self.inducer.forward(
                chars_var_inp,
                argmax=False,
                indices=indices,
                set_grammar=set_grammar
            )
        else:
            assert self.inducer.model_type == 'word'
            structure_loss = self.inducer.forward(
                word_inp,
                argmax=False,
                indices=indices,
                set_grammar=set_grammar
            )
        return structure_loss.sum().item()

