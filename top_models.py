import torch.nn as nn
import torch.distributions
from torch.utils.tensorboard import SummaryWriter
from treenode import convert_binary_matrix_to_strtree


DEBUG = True

def dprint(*args, **kwargs):
    if DEBUG: 
        print("DEBUG: ", end="")
        print(*args, **kwargs)


class TopModel(nn.Module):
    def __init__(self, inducer, writer:SummaryWriter=None):
        super(TopModel, self).__init__()
        self.inducer = inducer
        self.writer = writer


    def forward(self, word_inp, pos_inp, chars_var_inp, distance_penalty_weight=0.):
        if self.inducer.model_type == "char":
            logprob_list = self.inducer.forward(chars_var_inp, pos_inp, words=word_inp)
        else:
            assert self.inducer.model_type == 'word'
            logprob_list = self.inducer.forward(word_inp, pos_inp)

        total_num_chars = sum(
            [sum([x.numel() for x in y]) for y in chars_var_inp]
        )
        structure_loss = torch.sum(logprob_list, dim=0)

        total_loss = structure_loss
        return total_loss


    def parse(self, word_inp, pos_inp, chars_var_inp, indices, eval=False, set_grammar=True):
        if self.inducer.model_type == 'char':
            structure_loss, vtree_list, _, _ = self.inducer.forward(
                chars_var_inp,
                pos_inp,
                eval,
                argmax=True,
                indices=indices,
                set_grammar=set_grammar
            )
        else:
            assert self.inducer.model_type == 'word'
            structure_loss, vtree_list, _, _ = self.inducer.forward(
                word_inp,
                pos_inp,
                eval,
                argmax=True,
                indices=indices,
                set_grammar=set_grammar
            )
        return structure_loss.sum().item(), vtree_list


    def likelihood(self, word_inp, pos_inp, chars_var_inp, indices, set_grammar=True):
        if self.inducer.model_type == 'char':
            structure_loss = self.inducer.forward(
                chars_var_inp,
                pos_inp,
                argmax=False,
                indices=indices,
                set_grammar=set_grammar
            )
        else:
            assert self.inducer.model_type == 'word'
            structure_loss = self.inducer.forward(
                word_inp,
                pos_inp,
                argmax=False,
                indices=indices,
                set_grammar=set_grammar
            )
        return structure_loss.sum().item()

