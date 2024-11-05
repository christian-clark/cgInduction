import torch.nn as nn
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


    def forward(self, word_inp):
        loss = self.inducer.forward(word_inp)
        return loss.sum()


    def parse(self, word_inp, set_grammar=True):
        loss, vtree_list = self.inducer.forward(
            word_inp,
            argmax=True,
            set_grammar=set_grammar
        )
        return loss.sum().item(), vtree_list


    def likelihood(self, word_inp, set_grammar=True):
        loss = self.inducer.forward(
            word_inp,
            set_grammar=set_grammar
        )
        return loss.sum().item()

