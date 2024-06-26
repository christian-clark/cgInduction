import numpy as np
from nltk import tree
from collections import Counter, defaultdict


class Rule:
    def __init__(self, lhs, rhs1, rhs2=None):
        self.lhs = lhs
        self.rhs1 = rhs1
        self.rhs2 = rhs2
        if self.rhs2 is None:
            self.terminal = self.rhs1

    def get_rule(self):
        if self.rhs2 is not None:
            return (self.lhs, self.rhs1, self.rhs2)
        else:
            return (self.lhs, self.rhs1)


def nodes_to_tree(nodes, sent):
    sent_len = sent.shape[0]
    sent_words = list(range(sent_len))
    bracketed_string = [''] * (sent_len * 2 + 1 )
    bracketed_string[1::2] = [str(x) for x in sent_words]
    brackets = [''] * (sent_len+1)
    nodes.sort(key=lambda x : x.span_length, reverse=True)
    # print(nodes)
    for node in nodes:
        brackets[node.i] += '(' + str(node.k) + ' '
        brackets[node.j] = ')' + brackets[node.j]
    bracketed_string[::2] = brackets
    # print(bracketed_string)
    # exit()
    try:
        this_tree = tree.Tree.fromstring(''.join(bracketed_string))
    except:
        print(''.join(bracketed_string))
        print(nodes)
        raise
    productions = this_tree.productions()
    production_counter_dict = defaultdict(Counter)
    for rule in productions:
        production_counter_dict[rule.lhs()][rule.rhs()] += 1
    p0_counter = Counter()
    p0_counter[this_tree.label()] = 1
    production_counters = (production_counter_dict, p0_counter)
    l_branch, r_branch = calc_branching_score(this_tree)
    return this_tree, production_counters, (l_branch, r_branch)

class Node:
    def __init__(self, cat, cat_str, i, j, D=0, K=0, parent=None):
        # TODO get rid of unnecessary stuff here
        self.D = D
        self.K = K
        self.cat = int(cat)
        self.cat_str = cat_str
        self.i = i
        self.j = j
        self.s, self.d = -1, -1
        self.Q = (D+1) * 2 * K
        self.d = 0
        self.k = 0
        self.s = 0
        self.span_length = self.j - self.i
        self.parent = parent
        self.__unwind_cat()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "{} ({},{})".format((self.s, self.d, self.k), self.i, self.j)

    def str(self):
        return self.__str__()

    def __unwind_cat(self):
        if self.D == -1:
            #self.k = self.cat
            self.k = self.cat_str
        # TODO figure out what this case is for
        elif self.K != 0 and self.D != -1:
            self.s, self.d, self.k = np.unravel_index(self.cat, (2, self.D+1, self.K))
            # return "s{} d{} {}".format(self.s, self.d, self.k)
        else:
            return self.cat

    def is_terminal(self):
        if self.j - self.i == 1:
            return True
        return False


def calc_branching_score(t):
    r_branch = 0
    l_branch = 0
    # print(t)
    for position in t.treepositions():
        if not (isinstance(t[position],str) or isinstance(t[position][0],str)):
            if len(t[position][0]) == 2:
                l_branch += 1
            if len(t[position][1]) == 2:
                r_branch += 1
    return l_branch, r_branch

def convert_binary_matrix_to_strtree(binary_matrix, argmax_spans, argmax_tags, length, words):
    pred_span = [(a[0], a[1]) for a in argmax_spans[0]]
    binary_matrix = binary_matrix[0].cpu().numpy()
    label_matrix = np.zeros((length, length))
    for span in argmax_spans[0]:
        label_matrix[span[0]][span[1]] = span[2]
    pred_tree = {}
    for i in range(length):
        tag = "T-" + str(int(argmax_tags[i].item()) + 1)
        pred_tree[i] = "(" + str(tag) + " " + str(words[i]) + ")"
    for k in np.arange(1, length):
        for s in np.arange(length):
            t = s + k
            if t > length - 1: break
            if binary_matrix[s][t] == 1:
                nt = "NT-" + str(int(label_matrix[s][t]) + 1)
                span = "(" + nt + " " + pred_tree[s] + " " + pred_tree[t] + ")"
                pred_tree[s] = span
                pred_tree[t] = span
    pred_tree = pred_tree[0]
    pred_tree = tree.Tree.fromstring(pred_tree)
    return pred_tree
