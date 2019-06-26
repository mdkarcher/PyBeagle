import phyloinfer as pinf
# from loglikelihood import *
from gradient import *


# sample a random tree from the prior
n_tips = 5
true_tree = pinf.tree.create(n_tips, branch='random')
# true_tree.show()

data = pinf.data.treeSimu(true_tree, JC.D, JC.U, JC.U_inv, JC.pi, 100)

L = pinf.Loglikelihood.initialCLV(data)
true_branch = pinf.branch.get(true_tree)
true_llik = pinf.Loglikelihood.phyloLoglikelihood(true_tree, true_branch, JC.D, JC.U, JC.U_inv, JC.pi, L)
print(f"The log-likelihood of the true tree: {true_llik}")

true_grad = pinf.Loglikelihood.phyloLoglikelihood(true_tree, true_branch, JC.D, JC.U, JC.U_inv, JC.pi, L, grad=True)
np.round(true_grad, 1)

true_my_grad = gradient_loglikelihood(true_tree, data)
for i in range(len(true_my_grad)):
    print(f"Node {i}: ch={round(true_grad[i],1)}, mk={round(true_my_grad[i],1)}, diff={round(true_grad[i]-true_my_grad[i],9)}")

bg_grad = gradient_loglikelihood_beagle(true_tree, data)
for i in range(len(true_my_grad)):
    print(f"Node {i}: ch={round(true_grad[i],1)}, mk={round(bg_grad[i],1)}, diff={round(true_grad[i]-bg_grad[i],9)}")

bg_instance, tip_name_to_address = gradient_loglikelihood_beagle_init(data)
bg_grad_eval = gradient_loglikelihood_beagle_evaluate(bg_instance, true_tree, tip_name_to_address)
for i in range(len(true_my_grad)):
    print(f"Node {i}: ch={round(true_grad[i],1)}, mk={round(bg_grad_eval[i],1)}, diff={round(true_grad[i]-bg_grad_eval[i],9)}")
