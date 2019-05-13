import phyloinfer as pinf
import numpy as np
import random as rnd
import cProfile

from loglikelihood import *
from reroot import *
from tree_utils import *
from sequence_data import *
from models import *

# %run reroot.py
# %run loglikelihood.py
# %run tree_utils.py

# simulate Data
pden = np.array([.25,.25,.25,.25])

# decompose the rate matrix (JC model)
D, U, U_inv, rate_matrix = pinf.rateM.decompJC()

# sample a random tree from the prior
ntips = 15
rnd.seed(473291)
true_tree = pinf.tree.create(ntips, branch='random')
true_tree.show()

rnd.seed(290102)
data = pinf.data.treeSimu(true_tree, D, U, U_inv, pden, 100)

L = pinf.Loglikelihood.initialCLV(data)
true_branch = pinf.branch.get(true_tree)

true_llik = pinf.Loglikelihood.phyloLoglikelihood(true_tree, true_branch, D, U, U_inv, pden, L)
print(f"The log-likelihood of the true tree: {true_llik}")

# seq_enc = seq_encode(data)
# my_llik = loglikelihood(true_tree, seq_enc)
my_llik = loglikelihood(true_tree, data)
print(f"My log-likelihood of the true tree: {my_llik}")
round(my_llik - true_llik, 6)

tree_exp = true_tree.copy()
reroot(tree_exp, 14)
tree_exp.show()

exp_llik = loglikelihood(tree_exp, data)
print(f"Experimental log-likelihood of the true tree: {exp_llik}")
round(exp_llik - true_llik, 6)

tree_id = true_tree.copy()
tree_id.show()
refresh_ids(tree_id)

tree_id2 = tree_id.copy()
reroot(tree_id2, 4)
tree_id2.show()

tree_id3 = tree_id2.copy()
refresh_ids(tree_id3)
tree_id3.show()
loglikelihood(tree_id3, data)
loglikelihood(tree_id3, data, id_attr="id")
loglikelihood(tree_id3, data, id_attr="id", leaf_attr="name")

tree_rrt = true_tree.copy()
outgroup = tree_rrt.get_leaves()[10]
reroot(tree_rrt, outgroup)
refresh_ids(tree_rrt, attr="id")
rrt_llik = loglikelihood(tree_rrt, data, id_attr="id", leaf_attr="name")
print(f"Experimental log-likelihood of the true tree: {rrt_llik}")
round(rrt_llik - true_llik, 6)



# Below this line is out of date syntax
def test_lliks(n_taxa=15, n_sites=100):
    tree = pinf.tree.create(n_taxa, branch='random')
    branch = pinf.branch.get(tree)
    dna_data = pinf.data.treeSimu(tree, JC.D, JC.U, JC.U_inv, JC.pi, n_sites)
    L_cheng = pinf.Loglikelihood.initialCLV(dna_data)
    llik_cheng = pinf.Loglikelihood.phyloLoglikelihood(tree, branch, JC.D, JC.U, JC.U_inv, JC.pi, L_cheng)
    data_enc = seq_encode(dna_data)
    llik_mk = loglikelihood(tree, data_enc)
    rerooted = reroot(tree, rnd.choice(tree.get_leaf_names()))
    llik_rert = loglikelihood(rerooted, data_enc)
    return llik_mk - llik_cheng, llik_rert - llik_cheng


cProfile.run('test_lliks(50, 1000)')


def test_simu_cheng(n_taxa=15, n_sites=100):
    tree = pinf.tree.create(n_taxa, branch='random')
    branch = pinf.branch.get(tree)
    dna_data = pinf.data.treeSimu(tree, JC.D, JC.U, JC.U_inv, JC.pi, n_sites)


def test_simu_mk(n_taxa=15, n_sites=100):
    tree = pinf.tree.create(n_taxa, branch='random')
    seq = seq_sim(tree, n_sites)


cProfile.run('test_simu_cheng(50, 1000)')
cProfile.run('test_simu_mk(50, 1000)')



