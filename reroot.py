from ete3 import Tree
# import numpy as np
# import random as rnd
# import phyloinfer as pinf
import itertools


def find_old_root(tree, child1, child2, child3):
    # child1 = next(tree.iter_search_nodes(name=name1))
    # child2 = next(tree.iter_search_nodes(name=name2))
    # child3 = next(tree.iter_search_nodes(name=name3))
    for pair in itertools.combinations((child1, child2, child3), 2):
        common_ancestor = tree.get_common_ancestor(*pair)
        if common_ancestor not in pair:
            return common_ancestor


def exchange_features(node1, node2):
    for feature in (set(node1.features) | set(node2.features)) - {'dist', 'support'}:
        n1_val = None
        n2_val = None
        if feature in node1.features:
            n1_val = getattr(node1, feature)
        if feature in node2.features:
            n2_val = getattr(node2, feature)

        if n1_val is None:
            node2.del_feature(feature)
        else:
            node2.add_feature(feature, n1_val)

        if n2_val is None:
            node1.del_feature(feature)
        else:
            node1.add_feature(feature, n2_val)


def reroot(tree, out, id_attr="name"):
    if isinstance(out, Tree):
        outgroup=out
    else:
        search_dict = {id_attr: out}
        outgroup = next(tree.iter_search_nodes(**search_dict))
    if outgroup in tree.get_children():
        return tree
    root_children = tree.get_children()
    tree.set_outgroup(outgroup)
    old_root = find_old_root(tree, *root_children)
    exchange_features(tree, old_root)
    assert tree.name == ''
    assert len(tree.get_children()) == 2
    backup_child1, backup_child2 = tree.get_children()
    tree.unroot()
    # root_children = [child.name for child in tree.children]
    root_children = tree.get_children()
    assert (backup_child1 in root_children) ^ (backup_child2 in root_children)
    if backup_child1 in root_children:
        exchange_features(tree, backup_child2)
    else:
        exchange_features(tree, backup_child1)
    # outgroup = next(tree.iter_search_nodes(name=name))
    outgroup.dist *= 2
    # return tree

