import json, bidict, sys
from itertools import product as prod


DEBUG = False
def printDebug(*args, **kwargs):
    if DEBUG:
        print("DEBUG: ", end="")
        print(*args, **kwargs)


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
        elif other.is_primitive(): return False
        else:
            val1 = self.val
            val2 = other.val
            res1, arg1 = self.res_arg
            res2, arg2 = other.res_arg
            return val1 == val2 and res1 == res2 and arg1 == arg2

    # needed for building sets of CGNodes
    def __hash__(self):
        return hash(str(self))

    # needed for sorting lists of CGNodes
    def __lt__(self, other):
        return str(self) < str(other)



def generate_categories_by_depth(
    num_primitives, max_depth, max_arg_depth=None
):
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
    ix2depth = list()
    # sorting ensures cats will be added to ix2cat in a consistent order
    # across runs (for reproducibility)
    for cat in sorted(cs_dleq[0]):
        ix2cat[len(ix2cat)] = cat
        ix2depth.append(0)

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
        for cat in sorted(cs_deqiplus1):
            ix2cat[len(ix2cat)] = cat
            ix2depth.append(i+1)
    return cs_dleq, ix2cat, ix2depth


def category_from_string(string):
    printDebug("category:", string)
    OPERATORS = ["-a", "-b"]
    if string[0] == "{":
        assert string[-1] == "}"
        string = string[1:-1]
    return_cgnode = None

    # base case: string is primitive category
    primitive = True
    for op in OPERATORS:
        if op in string:
            primitive = False
            break
    if primitive:
        return_cgnode = CGNode(string)

    # recursive case: split string at the operator
    else:
        printDebug("not primitive")
        paren_count = 0
        split_ix = -1
        for ix, char in enumerate(string):
            if char == "{":
                paren_count += 1
            elif char == "}":
                paren_count -= 1
            if paren_count == 0:
                # find where the operator starts in the string
                # need to use find method bc there can be multi-character
                # primtiives like "10"
                # this assumes that all operators start with "-"
                split_ix = string.find("-", ix+1)
                break
        for op in OPERATORS:
            op_l = len(op)
            if string[split_ix:split_ix+op_l] == op:
                res = string[0:split_ix]
                arg = string[split_ix+op_l:]
                return_cgnode = CGNode(
                    op, category_from_string(res), category_from_string(arg)
                )
                break

    assert return_cgnode is not None
    return return_cgnode


def read_categories_from_file(f):
    all_cats = set()
    for l in open(f):
        cat = category_from_string(l.strip())
        if cat in all_cats:
            printDebug("warning: category {} is duplicated".format(cat))
        else:
            all_cats.add(cat)
    res_cats = set()
    arg_cats = set()
    for cat in all_cats:
        if cat.is_primitive(): continue
        res, arg = cat.res_arg
        if not res in all_cats or not arg in all_cats:
            raise Exception("if category (res)(op)(arg) is in the list, res and arg must be in the list too")
        res_cats.add(res)
        arg_cats.add(arg)
    # categories that can be arguments or results come first in ix2cat
    ix2cat = bidict.bidict()
    res_arg_cats = res_cats.union(arg_cats)
    for cat in res_arg_cats:
        ix2cat[len(ix2cat)] = cat
    for cat in all_cats - res_arg_cats:
        ix2cat[len(ix2cat)] = cat
    assert len(ix2cat) == len(all_cats)
    return all_cats, res_cats, arg_cats, ix2cat

