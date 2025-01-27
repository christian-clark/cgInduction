import torch.nn as nn
import torch.distributions

DEBUG = False

def printDebug(*args, **kwargs):
    if DEBUG:
        print("DEBUG: ", end="")
        print(*args, **kwargs)


class TopModel(nn.Module):
    def __init__(self, inducer):
        super(TopModel, self).__init__()
        self.inducer = inducer

    def forward(self, word_inp):
        logprob_list, _, _, _ = self.inducer.forward(word_inp)
        structure_loss = torch.sum(logprob_list, dim=0)
        return structure_loss

    def parse(self, word_inp, indices, eval=False, set_grammar=True):
        structure_loss, vtree_list, _, _ = self.inducer.forward(
            word_inp,
            eval,
            argmax=True,
            indices=indices,
            set_grammar=set_grammar
        )
        return structure_loss.sum().item(), vtree_list


    def likelihood(self, word_inp, indices, set_grammar=True):
        structure_loss = self.inducer.forward(
            word_inp,
            argmax=False,
            indices=indices,
            set_grammar=set_grammar
        )
        return structure_loss.sum().item()
