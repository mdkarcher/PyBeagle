import beagle as bg
import phyloinfer as pinf
import numpy as np

# simulate Data
pden = np.array([.25,.25,.25,.25])

# decompose the rate matrix (JC model)
D, U, U_inv, rate_matrix = pinf.rateM.decompJC()

# sample a random tree from the prior
ntips = 50
true_tree = pinf.tree.create(ntips, branch='random')

data = pinf.data.treeSimu(true_tree, D, U, U_inv, pden, 1000)

L = pinf.Loglikelihood.initialCLV(data)
true_branch = pinf.branch.get(true_tree)
print("The negative log-posterior of the true tree: {}".format(pinf.Logposterior.Logpost(true_tree, true_branch, D, U, U_inv, pden, L)))
print("The log-likelihood of the true tree: {}".format(pinf.Loglikelihood.phyloLoglikelihood(true_tree, true_branch, D, U, U_inv, pden, L)))

# sample a starting tree from the prior
init_tree = pinf.tree.create(ntips, branch='random')

# you may also want to take a look of the negative log-posterior and log-likelihood of the initial tree
init_branch = pinf.branch.get(init_tree)
print("The negative log-posterior of the init tree: {}".format(pinf.Logposterior.Logpost(init_tree, init_branch, D, U, U_inv, pden, L)))
print("The log-likelihood of the init tree: {}".format(pinf.Loglikelihood.phyloLoglikelihood(init_tree, init_branch, D, U, U_inv, pden, L)))
