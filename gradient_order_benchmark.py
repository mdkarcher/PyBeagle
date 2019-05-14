import phyloinfer as pinf
from models import *
from loglikelihood import *
from gradient import *

nsites = 100

ntips = 20
true_tree = pinf.tree.create(ntips, branch="random")
data = pinf.data.treeSimu(true_tree, JC.D, JC.U, JC.U_inv, JC.pi, nsites)

# %timeit grad = gradient_loglikelihood_beagle(true_tree, data)
# 2.99 ms ± 62.6 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

ntips = 40
true_tree = pinf.tree.create(ntips, branch="random")
data = pinf.data.treeSimu(true_tree, JC.D, JC.U, JC.U_inv, JC.pi, nsites)

# %timeit grad = gradient_loglikelihood_beagle(true_tree, data)
# 5.77 ms ± 44.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

ntips = 60
true_tree = pinf.tree.create(ntips, branch="random")
data = pinf.data.treeSimu(true_tree, JC.D, JC.U, JC.U_inv, JC.pi, nsites)

# %timeit grad = gradient_loglikelihood_beagle(true_tree, data)
# 8.6 ms ± 83.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

ntips = 80
true_tree = pinf.tree.create(ntips, branch="random")
data = pinf.data.treeSimu(true_tree, JC.D, JC.U, JC.U_inv, JC.pi, nsites)

# %timeit grad = gradient_loglikelihood_beagle(true_tree, data)
# 11.4 ms ± 95.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

ntips = 100
true_tree = pinf.tree.create(ntips, branch="random")
data = pinf.data.treeSimu(true_tree, JC.D, JC.U, JC.U_inv, JC.pi, nsites)

# %timeit grad = gradient_loglikelihood_beagle(true_tree, data)
# 14.5 ms ± 251 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

ntips = 200
true_tree = pinf.tree.create(ntips, branch="random")
data = pinf.data.treeSimu(true_tree, JC.D, JC.U, JC.U_inv, JC.pi, nsites)

# %timeit grad = gradient_loglikelihood_beagle(true_tree, data)
# 28.8 ms ± 421 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

ntips = 300
true_tree = pinf.tree.create(ntips, branch="random")
data = pinf.data.treeSimu(true_tree, JC.D, JC.U, JC.U_inv, JC.pi, nsites)

# %timeit grad = gradient_loglikelihood_beagle(true_tree, data)
# 44.5 ms ± 962 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

ntips = 500
true_tree = pinf.tree.create(ntips, branch="random")
data = pinf.data.treeSimu(true_tree, JC.D, JC.U, JC.U_inv, JC.pi, nsites)

# %timeit grad = gradient_loglikelihood_beagle(true_tree, data)
# 72.8 ms ± 906 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

ntips = 1000
true_tree = pinf.tree.create(ntips, branch="random")
data = pinf.data.treeSimu(true_tree, JC.D, JC.U, JC.U_inv, JC.pi, nsites)

# %timeit grad = gradient_loglikelihood_beagle(true_tree, data)
# 178 ms ± 950 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

ntips = 2000
true_tree = pinf.tree.create(ntips, branch="random")
data = pinf.data.treeSimu(true_tree, JC.D, JC.U, JC.U_inv, JC.pi, nsites)

# %timeit grad = gradient_loglikelihood_beagle(true_tree, data)
# 286 ms ± 7.21 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

ntips = 4000
true_tree = pinf.tree.create(ntips, branch="random")
data = pinf.data.treeSimu(true_tree, JC.D, JC.U, JC.U_inv, JC.pi, nsites)

# %timeit grad = gradient_loglikelihood_beagle(true_tree, data)
# 572 ms ± 8.08 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

xs = [20, 40, 60, 80, 100, 200, 300, 500, 1000, 2000, 4000]
ys = [2.99, 5.77, 8.6, 11.4, 14.5, 28.8, 44.5, 72.8, 178, 286, 572]
