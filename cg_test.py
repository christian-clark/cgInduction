from cg_type import CGTree, enumerate_structures, generate_labeled_trees
import bidict, torch

def test():
    t = CGTree()
    t.create_node("-b", "1")
    t.create_node("-b", "2", parent="1")
    t.create_node("-b", "3", parent="1")
    t.create_node("y", "4", parent="2")
    t.create_node("-b", "5", parent="2")
    t.create_node("y", "6", parent="3")
    t.create_node("y", "7", parent="3")
    t.create_node("y", "8", parent="5")
    t.create_node("y", "9", parent="5")

    t2 = CGTree()
    t2.create_node("-b", "1")
    t2.create_node("y", "2", parent="1")
    t2.create_node("-b", "3", parent="1")
    t2.create_node("y", "4", parent="3")
    t2.create_node("y", "5", parent="3")


    t3 = CGTree()
    t3.create_node("-b", "1")
    t3.create_node("y", "2", parent="1")
    t3.create_node("y", "3", parent="1")

    print("t:", t)
    print("t leaves:", t.leaves())
    print("t depth of node 9:", t.depth("9"))
    print("t depth of node 1:", t.depth("1"))
    print("t depth of node 4:", t.depth("4"))
    print("t2:", t2)
    print("t3:", t3)

    print("are t and t2 equal: {}".format(t == t2))
    print("are t.left_child and t2 equal: {}".format(t.subtree("2") == t2))

    print("t left apply t2: {}".format(t.left_apply(t2)))
    print("t left apply t3: {}".format(t.left_apply(t3)))
    print("t right apply t2: {}".format(t.right_apply(t2)))
    print("t right apply t3: {}".format(t.right_apply(t3)))


    t_clone = CGTree(t)

    print("t hash: {}".format(hash(t)))
    print("t2 hash: {}".format(hash(t2)))
    print("t_clone hash: {}".format(hash(t_clone)))



#for i in range(6):
    #print(len(enumerate_structures(i)))

MAX_DEPTH = 3
struct = enumerate_structures(MAX_DEPTH)
print("number of structures:", len(struct))
print(struct)

nt_options = ["-a", "-b"]
#t_options = ["0", "1", "2", "3"]
t_options = ["0", "1"]

all_trees = list()
tree_count = 0
for s in struct:
    #print("STRUCT:", s)
    lts = generate_labeled_trees(s, nt_options, t_options)
    print(len(lts))
    tree_count += len(lts)
    #for lt in lts: print(lt)
    #all_trees.extend(lts)

#print("total tree count:", len(all_trees))
print("total tree count:", tree_count)

#print(all_trees[::100])

ix2cat = bidict.bidict()
for t in all_trees:
    ix2cat[len(ix2cat)] = t

#print(list(ix2cat.items())[::100])


#children_ixs = list()
#parent_ixs = list()
#val_ixs = list()
#    
#for cat_ix in ix2cat:
#    cat = ix2cat[cat_ix]
#    kittens = cat.children(cat.root)
#    if len(kittens) == 0: continue
#    assert len(kittens) == 2
#    kitten1, kitten2 = kittens
#    kitten1 = cat.subtree(kitten1.identifier)
#    kitten1_ix = ix2cat.inverse[kitten1]
#    kitten2 = cat.subtree(kitten2.identifier)
#    kitten2_ix = ix2cat.inverse[kitten2]
#    root_tag = cat.get_node(cat.root).tag 
#    if root_tag == "-a":
#        children_ix = kitten2_ix*len(ix2cat) + cat_ix
#        children_ixs.append(children_ix)
#        parent_ixs.append(kitten1_ix)
#        val_ixs.append(1)
#    else:
#        assert root_tag == "-b"
#        children_ix = cat_ix*len(ix2cat) + kitten2_ix
#        children_ixs.append(children_ix)
#        parent_ixs.append(kitten1_ix)
#        val_ixs.append(1)


#rule_matrix = torch.sparse_coo_tensor(
#    [children_ixs, parent_ixs], val_ixs, (len(ix2cat)**2, len(ix2cat)
#))
#
#childrens, parents = rule_matrix.coalesce().indices()
#for i, children in enumerate(childrens):
#    children = children.item()
#    child1 = ix2cat[children // len(ix2cat)]
#    child2 = ix2cat[children % len(ix2cat)]
#    parent = ix2cat[parents[i].item()]
#    print('{}\t{}\t->\t{}'.format(child1, child2, parent))
