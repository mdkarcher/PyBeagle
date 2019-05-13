import phyloinfer as pinf
# import numpy as np
# import random as rnd
# import timeit

from loglikelihood import *
from reroot import *
from tree_utils import *
from sequence_data import *
from models import *


def cheng_lik_list(tree_list, seqs_list, model=JC):
    result = []
    for tree, seqs in zip(tree_list, seqs_list):
        L = pinf.Loglikelihood.initialCLV(seqs)
        branch = pinf.branch.get(tree)
        llik = pinf.Loglikelihood.phyloLoglikelihood(tree, branch, model.D, model.U, model.U_inv, model.pi, L)
        result.append(llik)
    return result


def mk_lik_list(tree_list, seqs_list, model=JC):
    result = []
    for tree, seqs in zip(tree_list, seqs_list):
        llik = loglikelihood(tree, seqs, model)
        result.append(llik)
    return result


def bg_lik_list(tree_list, seqs_list, model=JC):
    result = []
    for tree, seqs in zip(tree_list, seqs_list):
        llik = loglikelihood_beagle(tree, seqs, model, scaling=True)
        result.append(llik)
    return result


def bg_lik_list2(tree_list, seqs_list, model=JC):
    result = []
    for tree, seqs in zip(tree_list, seqs_list):
        llik = loglikelihood_beagle_rooting_test(tree, seqs, model, scaling=True)
        result.append(llik)
    return result

n_sims = 10
n_tips = 300
n_sites = 1500
model = JC

tree_list = []
seqs_list = []
for _ in range(n_sims):
    tree = pinf.tree.create(n_tips, branch='random')
    tree_list.append(tree)
    data = pinf.data.treeSimu(tree, model.D, model.U, model.U_inv, model.pi, n_sites)
    seqs_list.append(data)


cheng_lliks = cheng_lik_list(tree_list, seqs_list)
mk_lliks = mk_lik_list(tree_list, seqs_list)
bg_lliks = bg_lik_list(tree_list, seqs_list)

# timeit.timeit('cheng_lik_list(tree_list, seqs_list)', number=10)

# %timeit cheng_lliks = cheng_lik_list(tree_list, seqs_list)
# %timeit mk_lliks = mk_lik_list(tree_list, seqs_list)
# %timeit bg_lliks = bg_lik_list(tree_list, seqs_list)
# %timeit bg_lliks2 = bg_lik_list2(tree_list, seqs_list)


