from .compare_trees import main as sent_f1
from .evalb_unlabeled import eval_rvm_et_al
import tempfile
import nltk

def eval_access(pred_tree_list, gold_tree_list, epoch, section='dev'):

    gold_trees = []
    for t in gold_tree_list:
        gold_trees.append(nltk.tree.Tree.fromstring(t))

    p, r, f1, vm, rvm = eval_rvm_et_al((gold_trees, pred_tree_list))
