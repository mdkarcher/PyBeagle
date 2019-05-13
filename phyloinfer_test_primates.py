import numpy as np
import phyloinfer as pinf

# load primates data set
data, taxon = pinf.data.loadData('../datasets/primates.nex', 'nexus')

# initialize the tree
ntips = len(taxon)
init_tree = pinf.tree.create(ntips, branch='random')
init_branch = pinf.branch.get(init_tree)

# set the stationary frequency for JC model
pden = np.array([0.25, 0.25, 0.25, 0.25])

samp_res = pinf.phmc.hmc(init_tree, init_branch, (pden,1), data, 100, 0.004, 100,
                         subModel='JC', surrogate=True, burnin_frac=0.5, delta=0.008,
                         adap_stepsz_rate=0.8, printfreq=20)
