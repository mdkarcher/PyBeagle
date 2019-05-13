import phyloinfer as pinf
from loglikelihood import *


# sample a random tree from the prior
n_tips = 3
true_tree = pinf.tree.create(n_tips, branch='random')
true_tree.show()

data = pinf.data.treeSimu(true_tree, JC.D, JC.U, JC.U_inv, JC.pi, 100)

L = pinf.Loglikelihood.initialCLV(data)
true_branch = pinf.branch.get(true_tree)
true_llik = pinf.Loglikelihood.phyloLoglikelihood(true_tree, true_branch, JC.D, JC.U, JC.U_inv, JC.pi, L)
print(f"The log-likelihood of the true tree: {true_llik}")

true_grad = pinf.Loglikelihood.phyloLoglikelihood(true_tree, true_branch, JC.D, JC.U, JC.U_inv, JC.pi, L, grad=True)
np.round(true_grad, 1)

true_my_grad = gradient_loglikelihood(true_tree, data)
for i in range(len(true_my_grad)):
    print(f"Node {i}: ch={round(true_grad[i],1)}, mk={round(true_my_grad[i],1)}, diff={round(true_grad[i]-true_my_grad[i],1)}")

# sample a starting tree from the prior
init_tree = pinf.tree.create(n_tips, branch='random')

init_branch = pinf.branch.get(init_tree)
init_llik = pinf.Loglikelihood.phyloLoglikelihood(init_tree, init_branch, JC.D, JC.U, JC.U_inv, JC.pi, L)
print(f"The log-likelihood of the init tree: {init_llik}")

init_grad = pinf.Loglikelihood.phyloLoglikelihood(init_tree, init_branch, JC.D, JC.U, JC.U_inv, JC.pi, L, grad=True)
np.round(init_grad, 1)

init_my_grad = gradient_loglikelihood(init_tree, data)
for i in range(len(init_my_grad)):
    print(f"Node {i}: ch={round(init_grad[i],1)}, mk={round(init_my_grad[i],1)}, diff={round(init_grad[i]-init_my_grad[i],1)}")
