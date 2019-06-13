import phyloinfer as pinf
# import numpy as np
# import random as rnd
# import timeit

# from loglikelihood import *
from gradient import *
# from reroot import *
# from tree_utils import *
# from sequence_data import *
from models import *


def cheng_grad_list(tree_list, seqs_list, model=JC):
    result = []
    for tree, seqs in zip(tree_list, seqs_list):
        L = pinf.Loglikelihood.initialCLV(seqs)
        branch = pinf.branch.get(tree)
        grad = pinf.Loglikelihood.phyloLoglikelihood(tree, branch, model.D, model.U, model.U_inv, model.pi, L, grad=True)
        result.append(grad)
    return result


def mk_grad_list(tree_list, seqs_list, model=JC):
    result = []
    for tree, seqs in zip(tree_list, seqs_list):
        grad = gradient_loglikelihood(tree, seqs, model)
        result.append(grad)
    return result


def bg_grad_list(tree_list, seqs_list, model=JC):
    result = []
    for tree, seqs in zip(tree_list, seqs_list):
        grad = gradient_loglikelihood_beagle(tree, seqs, model)
        result.append(grad)
    return result


n_sims = 10
n_tips = 100
n_sites = 500
model = JC

tree_list = []
seqs_list = []
for _ in range(n_sims):
    tree = pinf.tree.create(n_tips, branch='random')
    tree_list.append(tree)
    data = pinf.data.treeSimu(tree, model.D, model.U, model.U_inv, model.pi, n_sites)
    seqs_list.append(data)


cheng_grads = cheng_grad_list(tree_list, seqs_list)
mk_grads = mk_grad_list(tree_list, seqs_list)
bg_grads = bg_grad_list(tree_list, seqs_list)

# timeit.timeit('cheng_lik_list(tree_list, seqs_list)', number=10)

for cheng, mk, bg in zip(cheng_grads, mk_grads, bg_grads):
    # for i in range(len(cheng)):
        # print(f"Cheng: {cheng[i]}, MK: {mk[i]}, Beagle: {bg[i]}")
        # print(f"MK_diff: {round(cheng[i] - mk[i], 6)}, BG_diff: {round(cheng[i] - bg[i], 6)}")
    print(all(abs(cheng[i] - bg[i]) < 0.000001 for i in range(len(cheng))))

# %timeit cheng_grad_list(tree_list, seqs_list)
# %timeit mk_grad_list(tree_list, seqs_list)
# %timeit bg_grad_list(tree_list, seqs_list)


