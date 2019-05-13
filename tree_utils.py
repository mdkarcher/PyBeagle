from ete3 import Tree

from models import *


def tree_length(tree):
    return sum(node.dist for node in tree.traverse())


def prob_trans(edge_length, model=JC):
    return model.U @ np.diag(np.exp(model.D * edge_length)) @ model.U_inv


def grad_trans(edge_length, model=JC):
    return model.U @ np.diag(model.D * np.exp(model.D * edge_length)) @ model.U_inv


def refresh_ids(tree, attr="id"):
    leaf_id = 0
    internal_id = len(tree)
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node.add_feature(attr, leaf_id)
            leaf_id += 1
        else:
            node.add_feature(attr, internal_id)
            internal_id += 1



