from itertools import product as prod


DEBUG = True
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

    def is_modifier(self):
        # u-av categories can be modifiers
        if self.is_primitive(): return False
        res, arg = self.res_arg
        return self.val == "-a" \
            and res.is_primitive() \
            and arg.is_primitive()

    def arg_depth(self):
        return CGNode._get_arg_depth(self)

    @staticmethod
    def _get_arg_depth(node):
        if node.is_primitive():
            return 0
        else:
            return 1 + CGNode._get_arg_depth(node.res_arg[0])

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
        # NOTE this previously set max_arg_depth to max_depth,
        # which I think was wrong
        max_arg_depth = max_depth - 1
    else:
        assert max_depth >= max_arg_depth
    # dictionary mapping integer i to set of cats of depth less than
    # or equal to i
    cs_dleq = dict()
    cs_dleq[-1] = {}
    cs_dleq[0] = {CGNode(str(p)) for p in range(num_primitives)}

    # categories that can be modifiers (must be u-av, where u and v
    # are primitives)
    mod_cats = set()
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
                new_cat = CGNode(o, res, arg)
                cs_deqiplus1.add(new_cat)
                if new_cat.is_modifier():
                    mod_cats.add(new_cat)
        cs_dleq[i+1] = cs_dleq[i].union(cs_deqiplus1)

    all_cats =  cs_dleq[max_depth]
    # categories that can be results from argument attachemnt
    res_cats = cs_dleq[max_depth-1]
    # categories that can be arguments to argument attachment
    # optionally constrain the complexity of argument categories
    arg_cats = cs_dleq[max_arg_depth]

    # categories that can be parents of a binary-branching rule
    par_cats = set()
    # types of categories that can be parents:
    # * u-av cats (modifiers can be modified)
    par_cats.update(mod_cats)
    # * res cats (can undergo argument attachment). A subset of these
    # can also be arguments and be modified
    par_cats.update(res_cats)

    # categories that can be generated children from a binary-branching rule
    gen_cats = set()
    # types of categories that can be generated children:
    # * u-av cats (modifiers)
    gen_cats.update(mod_cats)
    # * arg cats
    gen_cats.update(arg_cats)
    return all_cats, par_cats, gen_cats, arg_cats, res_cats


def category_from_string(string):
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
        elif cat.arg_depth() > 2:
            raise Exception("Inducer does not currently support categories that take more than 2 arguments")
        else:
            all_cats.add(cat)
    # categories that can be results from argument attachemnt
    res_cats = set()
    # categories that can be arguments to argument attachment
    arg_cats = set()
    # primitive categories
    prim_cats = set()
    # categories that can be modifiers (must be u-av, where u and v
    # are primitives)
    mod_cats = set()
    for cat in all_cats:
        if cat.is_primitive():
            prim_cats.add(cat)
            continue
        if cat.is_modifier():
            mod_cats.add(cat)
        res, arg = cat.res_arg
        if res in all_cats and arg in all_cats:
            res_cats.add(res)
            arg_cats.add(arg)
        # NOTE: hypothetically you could allow a category u-av to be included
        # even if u and v weren't in all_cats, since u-av can be a modifier.
        # But this complicates the logic enough in cg_inducer that I'm not
        # doing it
        else:
            raise Exception("if category (res)(op)(arg) is in the list, and it can't be a modifier category, res and arg must be in the list too.")
    # categories that can be parents of a binary-branching rule
    par_cats = set()
    # types of categories that can be parents:
    # * u-av cats (modifiers can be modified)
    par_cats.update(mod_cats)
    # * primitive cats (can be modified)
    par_cats.update(prim_cats)
    # * arg cats (can be modified)
    par_cats.update(arg_cats)
    # * res cats (can undergo argument attachment)
    par_cats.update(res_cats)

    # categories that can be generated children from a binary-branching rule
    gen_cats = set()
    # types of categories that can be generated children:
    # * u-av cats (modifiers)
    gen_cats.update(mod_cats)
    # * arg cats
    gen_cats.update(arg_cats)

    return all_cats, par_cats, gen_cats, arg_cats, res_cats


# replaced by CGNode.arg_depth
#def arg_depth(category):
#    """Return the number of arguments a syntactic category needs to result
#    in a primitive. Modificands (e.g. combining with 0/0) are not treated as
#    arguments."""
#    if category.is_primitive():
#        return 0
#    else:
#        res, arg = category.res_arg
#        # TODO this deals with modifiers, so that e.g. 0/0 will still have
#        # depth 0. Once the inducer includes a separate operation for
#        # modification, this special case should probably be removed
#        if res == arg:
#            return arg_depth(res)
#        else:
#            return 1 + arg_depth(res)
#
#
#def get_category_argument_depths(ix2cat):
#    ix2argdepth = dict()
#    for i, cat in ix2cat.items():
#        ix2argdepth[i] = arg_depth(cat)
#    return ix2argdepth