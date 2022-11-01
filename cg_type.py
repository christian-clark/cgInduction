from treelib import Tree, Node
import itertools, json


class CGTree(Tree):
    def __str__(self):
        node_stack = list()
        out = list()
        if self.root is None: return "<EMPTY>"
        CGTree.build_str(self, self.root, node_stack, out)
        return "".join(out)

    @staticmethod
    def build_str(tree, node, stack, out):
        children = tree.children(node)
        if len(children) == 0:
            out.append(tree.get_node(node).tag)
        else:
            # must be binary tree
            assert len(children) == 2
            stack.append(tree.get_node(node).tag)
            out.append("{")
            CGTree.build_str(tree, children[0].identifier, stack, out)
            out.append(stack.pop())
            CGTree.build_str(tree, children[1].identifier, stack, out)
            out.append("}")

    def __repr__(self):
        return str(self)
        
    def __eq__(self, other):
        root1 = self.get_node(self.root)
        root2 = other.get_node(other.root)
        children1 = self.children(self.root)
        children2 = other.children(other.root)

        # base case: two leaf nodes
        # for some reason doing root1.is_leaf() and root2.is_leaf() doesn't
        # work for this
        #if root1.is_leaf() and root2.is_leaf():
        if len(children1) == 0 and len(children2) == 0:
            if root1.tag == root2.tag: return True
            else: return False

        # recursive case: compare all pairs of children
        else:
            children1 = self.children(self.root)
            children2 = other.children(other.root)
            if len(children1) != len(children2): return False
            for i, child1 in enumerate(children1):
                child1 = self.subtree(child1.identifier)
                child2 = other.subtree(children2[i].identifier)
                if not child1 == child2: return False
        return True

    # needed for building sets of CGTrees
    def __hash__(self):
        return hash(str(self))

    def left_apply(self, other):
        """
        Attempts to combine self with other, when other is to the left of
        self. Self is used as the functor and other as the argument. If self
        and other are able to combine, returns the result. Otherwise returns
        None.
        
        Examples:
        N V-aN -> V
        V V-aN -> None
        """
        if self.get_node(self.root).tag != "-a": return None
        children = self.children(self.root)
        if not children: return None
        children = [self.subtree(c.identifier) for c in children]
        assert len(children) == 2
        if children[1] == other: return children[0]
        return None

    def right_apply(self, other):
        """
        Like left_apply except other is to the right of self.

        Examples:
        N V-aN-bN -> V-aN
        V V-aN-bN -> None
        """
        if self.get_node(self.root).tag != "-b": return None
        children = self.children(self.root)
        if not children: return None
        children = [self.subtree(c.identifier) for c in children]
        assert len(children) == 2
        if children[1] == other: return children[0]
        return None


    def get_argument(self):
        """
        Assuming self is a complex category, returns the argument category
        that self needs.
        e.g. get_argument(X-aY-bZ) -> Z

        If self is a primitive category, returns None.
        """
        # self.children(self.root)[0] is the resulting category after
        # self combines with argument
        children = self.children(self.root)
        if len(children) == 0: return None
        assert len(children) == 2
        return self.subtree(children[1].identifier)
        

def enumerate_structures(max_depth):
    start_tree = CGTree()
    start_tree.create_node("x")
    all_trees = {start_tree}
    candidates = {start_tree}
    while len(candidates) > 0:
        curr = candidates.pop()
        leaves = curr.leaves()
        for leaf in leaves:
            lid = leaf.identifier
            depth = curr.depth(lid)
            if depth < max_depth:
                new = CGTree(curr)
                # add left and right children to the leaf
                new.create_node("x", parent=lid)
                new.create_node("x", parent=lid)
                all_trees.add(new)
                candidates.add(new)
    return all_trees


def generate_labeled_trees(structure, nt_options, t_options):
    nodes = list()
    per_node_label_options = list()
    for n in structure.all_nodes():
        nid = n.identifier
        children = structure.children(nid)
        if len(children) > 0:
            per_node_label_options.append(nt_options)
        else:
            per_node_label_options.append(t_options)
        nodes.append(nid)

    all_labeled_trees = list()
    for label_assignment in itertools.product(*per_node_label_options):
        labeled_tree = CGTree(structure, deep=True)
        for i, asgmt in enumerate(label_assignment):
            nid = nodes[i]
            labeled_tree.get_node(nid).tag = asgmt
        all_labeled_trees.append(labeled_tree)

    return all_labeled_trees


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

