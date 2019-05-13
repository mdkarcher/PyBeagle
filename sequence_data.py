import random as rnd

from models import *
from tree_utils import *

dna_enc = {'A': [1., 0., 0., 0.], 'G': [0., 1., 0., 0.], 'C': [0., 0., 1., 0.], 'T': [0., 0., 0., 1.],
           'a': [1., 0., 0., 0.], 'g': [0., 1., 0., 0.], 'c': [0., 0., 1., 0.], 't': [0., 0., 0., 1.],
           '0': [1., 0., 0., 0.], '1': [0., 1., 0., 0.], '2': [0., 0., 1., 0.], '3': [0., 0., 0., 1.]}
dna_default = [1., 1., 1., 1.]

dna_ids = {'A': 0, 'G': 1, 'C': 2, 'T': 3,
           'a': 0, 'g': 1, 'c': 2, 't': 3,
           '0': 0, '1': 1, '2': 2, '3': 3}
dna_id_default = 4



def sample_root(n_sites, model=JC):
    return rnd.choices(range(len(model.pi)), weights=model.pi, k=n_sites)


def sample_state(parent_state, edge_length, model=JC):
    Pt = prob_trans(edge_length, model)
    result = parent_state.copy()
    for i, par in enumerate(parent_state):
        result[i] = rnd.choices(range(len(model.pi)), weights=Pt[par], k=1)[0]
    return result


# Makes a dictionary of lists of integers representing sequence data
def seq_sim(tree, n_sites, model=JC, id_attr=None, leaf_attr=None):
    if id_attr is None:
        id_attr = "name"
    if leaf_attr is None:
        leaf_attr = id_attr
    assert len(tree) >= 3
    result = dict()
    internal_state = dict()
    for node in tree.traverse("preorder"):
        if node.is_root():
            internal_state[getattr(node, id_attr)] = sample_root(n_sites, model)
        elif node.is_leaf():
            result[getattr(node, leaf_attr)] = sample_state(internal_state[getattr(node.up, id_attr)],
                                                            node.dist, model)
        else:
            internal_state[getattr(node, id_attr)] = sample_state(internal_state[getattr(node.up, id_attr)],
                                                                  node.dist, model)
    return result


# Convert from matrix of characters or ints to list of indicator matrices
def seq_encode(data):
    n_taxa, n_sites = data.shape
    encoded = [None] * n_taxa
    for i in range(n_taxa):
        encoded[i] = np.transpose([dna_enc.get(c, dna_default) for c in data[i]])
    return encoded


# Convert from dictionary of lists to dictionary of lists of indicator matrices
def dict_lists_to_dict_mats(data):
    result = dict()
    for key in data:
        result[key] = np.transpose([dna_enc.get(c, dna_default) for c in data[key]])
    return result


def arraylike_to_dict_mats(data):
    result = dict()
    for row in range(len(data)):
        result[row] = np.transpose([dna_enc.get(c, dna_default) for c in data[row]])
    return result


def convert_to_dict_mats(data):
    result = dict()
    if isinstance(data, dict):
        keys = data.keys()
    else:
        keys = range(len(data))
    for key in keys:
        result[key] = np.transpose([dna_enc.get(c, dna_default) for c in data[key]])
    return result


def convert_to_dict_lists(data):
    result = dict()
    if isinstance(data, dict):
        keys = data.keys()
    else:
        keys = range(len(data))
    for key in keys:
        result[key] = [dna_ids.get(c, dna_id_default) for c in data[key]]
    return result


