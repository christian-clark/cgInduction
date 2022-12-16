import json, bidict
from itertools import product as prod


class CGNode:
    def __init__(self, val=None, res=None, arg=None):
        if val is None:
            self.val = "x"
        else:
            self.val = val
        if res is None:
            assert arg is None
            self.res_arg = None
        else:
            assert arg is not None
            self.res_arg = (res, arg)

    def is_primitive(self):
        return self.res_arg is None

    def __str__(self):
        stack, out = list(), list()
        CGNode._build_str(self, stack, out)
        return "".join(out)

    @staticmethod
    def _build_str(node, stack, out):
        if node.is_primitive():
            out.append(node.val)
        else:
            res, arg = node.res_arg
            stack.append(node.val)
            out.append("{")
            CGNode._build_str(res, stack, out)
            out.append(stack.pop())
            CGNode._build_str(arg, stack, out)
            out.append("}")

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if self.is_primitive():
            return other.is_primitive() and self.val == other.val
        else:
            res1, arg1 = self.res_arg
            res2, arg2 = other.res_arg
            return res1 == res2 and  arg1 == arg2

    # needed for building sets of CGNodes
    def __hash__(self):
        return hash(str(self))




def generate_categories(num_primitives, max_depth, max_arg_depth=None):
    OPERATORS = ["-a", "-b"]
    if max_arg_depth is None:
        max_arg_depth = max_depth
    else:
        assert max_depth >= max_arg_depth
    # dictionary mapping integer i to set of cats of depth less than
    # or equal to i
    cs_dleq = dict()
    cs_dleq[-1] = {}
    cs_dleq[0] = {CGNode(str(p)) for p in range(num_primitives)}
    ix2cat = bidict.bidict()
    for cat in cs_dleq[0]:
        ix2cat[len(ix2cat)] = cat

    for i in range(max_depth):
        cs_dleqi = cs_dleq[i]
        # constrain argument cat to have depth no greater than max_arg_depth
        cs_arg = cs_dleq[min([i, max_arg_depth])]
        cs_dleqiminus1 = cs_dleq[i-1]
        children_i = set(prod(cs_dleqi, cs_arg))
        children_iminus1 = set(prod(cs_dleqiminus1, cs_dleqiminus1))
        cs_deqiplus1 = set()
        # children_i - children_iminus1 is the set of (result, argument)
        # pairs such that the result or argument is of depth i. When
        # a parent node is created with these two children, its depth will
        # be i+1
        for res, arg in children_i - children_iminus1:
            for o in OPERATORS:
                cs_deqiplus1.add(CGNode(o, res, arg))
        cs_dleq[i+1] = cs_dleq[i].union(cs_deqiplus1)
        for cat in cs_deqiplus1:
            ix2cat[len(ix2cat)] = cat
    return cs_dleq, ix2cat


# TODO implement this and probably remove trees_from_json and
# _recursive_build_tree
def cgnode_from_string(string):
    pass


def trees_from_json(cats_json):
    all_trees = list()
    j = json.load(open(cats_json))
    for jtree in j:
        tree = CGTree()
        _recursive_build_tree(tree, jtree)
        all_trees.append(tree)
    return all_trees


def _recursive_build_tree(tree, jtree, parent=None):
    new = jtree[0]
    new_node = Node(new)
    tree.add_node(new_node, parent=parent)
    if len(jtree) == 3:
        lchild = jtree[1]
        rchild = jtree[2]
        nid = new_node.identifier
        _recursive_build_tree(tree, lchild, parent=nid)
        _recursive_build_tree(tree, rchild, parent=nid)
    else:
        assert len(jtree) == 1

