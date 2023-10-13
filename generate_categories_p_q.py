import argparse, bidict, math, sys, json
from itertools import product as prod
from queue import PriorityQueue
from itertools import chain, combinations
from copy import deepcopy
from configparser import ConfigParser

# ways to mutate a category to form more categories:
# - branch at a subset of maximum-depth leaf nodes. only nodes of category
#   zero can branch.
#   each new leaf node from the branching will be category 0
# - increment a subset of primitives that are at the maximum-depth leaves.
#   only the highest primitives at this depth can be incremented. for
#   instance, if the max-depth leaf nodes currently have primitives
#   0, 2, and 1, respectively, only the 2 node can be incremented up to 3

# this way of generating categories means that a given complex category
# has only one possible ancestor category


DEFAULT_CONFIG = {
    "DEFAULT": {
        "p": 0.5,
        "q": 0.5,
        "numCats": 100,
        "minLogProb": -100,
        #"maxCost": 10,
        #"noBranchCost": 1,
        #"categoryCost": {
        #    "mode": "linear",
        #    "slope": 1,
        #    "intercept": 0
        #},
        "opACost": 0,
        "opBCost": 0
    }
}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


#https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
def powerset_minus_empty(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


class CategoryTree:
    # TODO prim_options should be passed in through config somehow
    def __init__(self, prim=None, op=None, res=None, arg=None,
        op_options={"-a", "-b"}):
        self.op_options = op_options
        if prim is not None:
            self._initialize_primitive(prim)
        else:
            assert op is not None 
            assert res is not None 
            assert arg is not None
            self._initialize_complex(op, res, arg)


    def _initialize_primitive(self, primitive):
        self.depth = 0
        assert primitive not in self.op_options
        self.array = [primitive]
        self.initilialized = True


    def _initialize_complex(self, operator, result, argument):
        self.depth = max(result.depth, argument.depth) + 1
        array_len = 2**(self.depth+1) - 1
        self.array = [None] * array_len
        assert operator in self.op_options
        assert result.op_options == argument.op_options
        self.array[0] = operator
        self._merge_arrays(result, argument)
        self.initilialized = True


    def _merge_arrays(self, result, argument):
        """
        Merge the array representations of the result and argument
        CategoryTree objects into the new CategoryTree
        """
        curr_ix = 0
        max_depth = max(result.depth, argument.depth)
        for i in range(max_depth+1):
            res_offset = 2**i
            arg_offset = 2**(i+1)
            for j in range(2**i):
                if i <= result.depth:
                    self.array[curr_ix+res_offset] = result.array[curr_ix]
                if i <= argument.depth:
                    self.array[curr_ix+arg_offset] = argument.array[curr_ix]
                curr_ix += 1


    def is_primitive(self, index=0):
        val = self.array[index]
        return val is not None and val not in self.op_options


    def replace_leaf(self, index, new_tree):
        assert self.is_primitive(index)
        leaf_depth = math.floor(math.log2(index+1))
        new_depth = new_tree.depth + leaf_depth
        self._increase_depth_if_needed(new_depth)
        curr_ix = 0
        for i in range(new_tree.depth+1):
            offset = index * (2**i)
            for j in range(2**i):
                self.array[curr_ix+offset] = new_tree.array[curr_ix]
                curr_ix += 1

 
    def _increase_depth_if_needed(self, new_depth):
        if new_depth <= self.depth:
            return
        else:
            new_array_len = 2**(new_depth+1) - 1
            extension = [None] * (new_array_len-len(self))
            self.array.extend(extension)
            self.depth = new_depth


    def get_max_depth_leaves(self):
        leaf_min_ix = 2**self.depth - 1
        leaf_max_ix = 2**(self.depth+1) - 1
        leaf_vals = list()
        leaf_ixs = list()
        for i in range(leaf_min_ix, leaf_max_ix):
            val = self.array[i]
            if val is not None:
                assert self.is_primitive(i)
                leaf_vals.append(val)
                leaf_ixs.append(i)
        return leaf_vals, leaf_ixs


    def get_max_depth_maximal_primitives(self):
        leaf_vals, leaf_ixs = self.get_max_depth_leaves()
        max_prim = max(leaf_vals)
        max_prim_ixs = [i for i, j in enumerate(leaf_vals) if j == max_prim]
        array_ixs = [leaf_ixs[i] for i in max_prim_ixs]
        return max_prim, array_ixs


    def get_max_depth_zero_primitives(self):
        leaf_vals, leaf_ixs = self.get_max_depth_leaves()
        zero_ixs = [i for i, j in enumerate(leaf_vals) if j == 0]
        array_ixs = [leaf_ixs[i] for i in zero_ixs]
        return array_ixs


    def __str__(self):
        stack, out = list(), list()
        self._build_str(0, stack, out)
        return "".join(out)


    def _build_str(self, index, stack, out):
        value = self.array[index]
        if self.is_primitive(index):
            out.append(str(value))
        else:
            res_ix = index*2 + 1
            arg_ix = index*2 + 2
            stack.append(value)
            out.append("{")
            self._build_str(res_ix, stack, out)
            out.append(stack.pop())
            self._build_str(arg_ix, stack, out)
            out.append("}")


    def __len__(self):
        return len(self.array)


    def __repr__(self):
        return str(self)


    def __eq__(self, other):
        return self._subtree_equal(other, 0)


    def _subtree_equal(self, other, index):
        if self.array[index] == other.array[index]:
            # base case: primitives at leaves
            if self.is_primitive(index):
                return True
            else:
                res_ix = index*2 + 1
                arg_ix = index*2 + 2
                return self._subtree_equal(other, res_ix) \
                    and self._subtree_equal(other, arg_ix)
        else:
            return False


    # needed for building sets
    def __hash__(self):
        return hash(str(self))


def cat_cost(cat_id, config):
    config = config["categoryCost"].replace("'", '"')
    config = json.loads(config)
    if config["mode"] == "linear":
        m = config["slope"]
        b = config["intercept"]
        return m*cat_id + b
    else:
        return cat_id


def generate_categories(config):
    num_cats = config.getint("numCats")
    # use negative log probs: smaller score means more likely
    max_score = -1 * config.getfloat("minLogProb")
    # generate a primitive
    p = -1 * math.log(config.getfloat("p"))
    # generate an argument category
    not_p = -1 * math.log(1-config.getfloat("p"))
    # stick with the current primitive
    q = -1 * math.log(config.getfloat("q"))
    # increment the current primitive
    not_q = -1 * math.log(1-config.getfloat("q"))

    categories, scores = list(), list()
    queue = PriorityQueue()
    # put_index is for breaking ties in the priority queue
    put_index = 0
    start_t = CategoryTree(prim=0)
    # score for category 0 (p for choosing a primitive, q for not
    # moving to next primitive)
    start_score = p + q
    queue.put((start_score, put_index, start_t))
    put_index += 1
    split_ta = CategoryTree(op="-a", res=start_t, arg=start_t)
    split_tb = CategoryTree(op="-b", res=start_t, arg=start_t)
    while len(categories) < num_cats: #and not queue.empty():
        score, _, t = queue.get_nowait()
        if score > max_score:
            break

        # increment max_depth maximal primitives
        curr_max_prim, tree_ixs = t.get_max_depth_maximal_primitives()
        for indices in powerset_minus_empty(tree_ixs):
            t_copy_incr = deepcopy(t)
            new_prim = curr_max_prim + 1
            new_prim_t = CategoryTree(prim=new_prim)
            t_score_incr = score
            # score change for replacing a single category k with a 
            # category k+1
            score_change = not_q
            for ix in indices:
                t_copy_incr.replace_leaf(ix, new_prim_t)
                t_score_incr += score_change
            queue.put((t_score_incr, put_index, t_copy_incr))
            put_index += 1

        # split max-depth zero primitives
        zero_ixs = t.get_max_depth_zero_primitives()
        for indices in powerset_minus_empty(zero_ixs):
            t_copy_split_a = deepcopy(t)
            t_copy_split_b = deepcopy(t)
            # (net) score change for replacing a noBranch with a branch and
            # adding two noBranches each with category 0
            score_change = not_p + p + q
            t_score_split = score
            for ix in indices:
                t_copy_split_a.replace_leaf(ix, split_ta)
                t_copy_split_b.replace_leaf(ix, split_tb)
                t_score_split += score_change
            queue.put((t_score_split, put_index, t_copy_split_a))
            put_index += 1
            queue.put((t_score_split, put_index, t_copy_split_b))
            put_index += 1

        categories.append(t)
        scores.append(score)
    return categories, scores


def main():
    top_config = ConfigParser()
    top_config.read_dict(DEFAULT_CONFIG)
    if len(sys.argv) == 1:
        overrides = []
    elif len(sys.argv[1].split("=")) == 1:
        if sys.argv[1] in {"-h", "--help"}:
            print('Usage: {0} [config] [overrides]'.format(sys.argv[0]), file=sys.stderr)
            return
        top_config.read(sys.argv[1])
        overrides = sys.argv[2:]
    else:
        overrides = sys.argv[1:]
    config = top_config["DEFAULT"]
    # any args after the config file override key-value pairs
    for kv in overrides:
        k, v = kv.split("=")
        config[k] = v
    
    categories, costs = generate_categories(config)
    print("Category\tCost")
    for i, cat in enumerate(categories):
        cost = costs[i]
        print("{}\t{}".format(cat, round(cost, 2)))


if __name__ == "__main__":
    main()

