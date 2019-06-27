import phyloinfer as pinf

from gradient import *
from models import *


def cheng_grad_list_eval(tree_list, branch_list, L, model=JC):
    result = []
    for tree, branch in zip(tree_list, branch_list):
        # branch = pinf.branch.get(tree)
        grad = pinf.Loglikelihood.phyloLoglikelihood(tree, branch, model.D, model.U, model.U_inv, model.pi, L, grad=True)
        result.append(grad)
    return result


def bg_grad_list_eval(tree_list, bg_init, model=JC):
    bg_instance, tip_name_to_address = bg_init
    result = []
    for tree in tree_list:
        grad = gradient_loglikelihood_beagle_evaluate(bg_instance, tree, tip_name_to_address, scaling=False)
        result.append(grad)
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
bg_init = gradient_loglikelihood_beagle_init(data, model, scaling=False)
for _ in range(n_sims):
    tree = pinf.tree.create(n_tips, branch='random')
    tree_list.append(tree)
    branch = pinf.branch.get(tree)
    branch_list.append(branch)

cheng_grads = cheng_grad_list_eval(tree_list, branch_list, L, model)
bg_grads = bg_grad_list_eval(tree_list, bg_init, model)
for i, grad in enumerate(cheng_grads):
    print(f"Tree:{i}")
    for j, deriv in enumerate(grad):
        print(f"    Edge {j}: ch={round(cheng_grads[i][j], 1)}, mk={round(bg_grads[i][j], 1)}, diff={round(cheng_grads[i][j] - bg_grads[i][j], 9)}")

# %timeit cheng_grad_list_eval(tree_list, branch_list, L, model)
# %timeit bg_grad_list_eval(tree_list, bg_init, model)
