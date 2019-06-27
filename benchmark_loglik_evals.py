import phyloinfer as pinf

from loglikelihood import *
from models import *


def cheng_lik_list_eval(tree_list, branch_list, L, model=JC):
    result = []
    for tree, branch in zip(tree_list, branch_list):
        # branch = pinf.branch.get(tree)
        llik = pinf.Loglikelihood.phyloLoglikelihood(tree, branch, model.D, model.U, model.U_inv, model.pi, L)
        result.append(llik)
    return result


def bg_lik_list_eval(tree_list, bg_init, model=JC):
    bg_instance, tip_name_to_address = bg_init
    result = []
    for tree in tree_list:
        llik = loglikelihood_beagle_evaluate(bg_instance, tree, tip_name_to_address, scaling=True)
        result.append(llik)
    return result


n_sims = 10
n_tips = 200
n_sites = 1000
model = JC

tree_list = []
branch_list = []
true_tree = pinf.tree.create(n_tips, branch='random')
data = pinf.data.treeSimu(true_tree, model.D, model.U, model.U_inv, model.pi, n_sites)
L = pinf.Loglikelihood.initialCLV(data)
bg_init = loglikelihood_beagle_init(data, model, scaling=True)
for _ in range(n_sims):
    tree = pinf.tree.create(n_tips, branch='random')
    tree_list.append(tree)
    branch = pinf.branch.get(tree)
    branch_list.append(branch)

cheng_lliks = cheng_lik_list_eval(tree_list, branch_list, L, model)
bg_lliks = bg_lik_list_eval(tree_list, bg_init, model)
np.round(np.array(cheng_lliks) - np.array(bg_lliks), 9)

# %timeit cheng_lik_list_eval(tree_list, branch_list, L, model)
# %timeit bg_lik_list_eval(tree_list, bg_init, model)
