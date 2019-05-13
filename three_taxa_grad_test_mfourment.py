import beagle as bg
import phyloinfer as pinf
import numpy as np
import random as rnd

#from models import *
#from loglikelihood import *
from phyloinfer import dataManipulation
from phyloinfer import treeManipulation
from phyloinfer import rateMatrix
from phyloinfer import Loglikelihood
from phyloinfer import branchManipulation

ntips = 40
nsites = 10
true_tree = treeManipulation.create(ntips, branch="random")
print(true_tree)
#true_tree.show()

D_JC, U_JC, U_JC_inv, rate_matrix_JC = pinf.rateMatrix.decompJC()
data = pinf.dataManipulation.treeSimu(true_tree, D_JC, U_JC, U_JC_inv, np.array([0.25]*4), nsites)

L = pinf.Loglikelihood.initialCLV(data)
true_branch = pinf.branchManipulation.get(true_tree)
print(true_branch)
true_llik = pinf.Loglikelihood.phyloLoglikelihood(true_tree, true_branch, D_JC, U_JC, U_JC_inv, np.array([0.25]*4), L)
print(f"The log-likelihood of the true tree: {true_llik}")

my_llik = 0 #loglikelihood(true_tree, data)
print(f"My log-likelihood of the true tree: {my_llik}")
round(my_llik - true_llik, 6)

true_grad = pinf.Loglikelihood.phyloLoglikelihood(true_tree, true_branch, D_JC, U_JC, U_JC_inv, np.array([0.25]*4), L, grad=True)
np.round(true_grad, 6)
print(true_grad)

# true_my_grad = gradient_loglikelihood(true_tree, data)
# for i in range(len(true_my_grad)):
#     print(f"Node {i}: ch={round(true_grad[i], 6)}, mk={round(true_my_grad[i], 6)}, diff={round(true_grad[i]-true_my_grad[i], 6)}")


# Beagle section

tree = true_tree.copy()
id_attr = "id"
leaf_attr = "name"

# Set up

table = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
         'a': 0, 'c': 1, 'g': 2, 't': 3,
         0: 0, 1: 1, 2: 2, 3: 3, '-': 4}

n_taxa, n_patterns = data.shape
n_states = 4

n_internals = n_taxa - 2
n_edges = 2*n_taxa - 3

n_partials = n_internals + n_edges
n_transition_probs = n_edges
n_derivatives = n_edges
n_matrices = n_transition_probs + n_derivatives

#outgroup = next(tree.iter_search_nodes(name=0))
outgroup = tree.children[0]
#refresh_ids(tree, attr=id_attr)
index = n_taxa
for node in tree.traverse('postorder'):
    if node.is_leaf():
        node.id = int(node.name)
    else:
        node.id = index
        index += 1
# Sanity check(s)

assert set(getattr(node, id_attr) for node in tree.get_leaves()) == set(range(len(tree)))
assert set(getattr(node, id_attr) for node in tree.traverse()) == set(range(2*len(tree)-2))

# Instantiate Beagle

requirementFlags = 0
# requirementFlags = bg.BEAGLE_FLAG_PRECISION_DOUBLE | bg.BEAGLE_FLAG_SCALING_MANUAL
# requirementFlags = bg.BEAGLE_FLAG_SCALING_MANUAL

returnInfo = bg.BeagleInstanceDetails()
instance = bg.beagleCreateInstance(
    n_taxa,             # tips
    # n_internals,        # partials
    n_partials, # partials
    n_taxa,             # sequences
    n_states,           # states
    n_patterns,         # patterns
    1,                  # models
    # n_transition_probs, # transition matrices
    n_matrices, # transition matrices
    1,                  # rate categories
    0,    # scale buffers
    # n_internals + 1,    # scale buffers
    None,               # resource restrictions
    0,                  # length of resource list
    0,                  # preferred flags
    requirementFlags,   # required flags
    returnInfo          # output details
)

for node in tree.get_leaves():
    states = bg.createStates(data[getattr(node, leaf_attr)], table)
    bg.beagleSetTipStates(instance, getattr(node, id_attr), states)

# For now, let all sites have equal weight
patternWeights = bg.createPatternWeights([1]*n_patterns)
bg.beagleSetPatternWeights(instance, patternWeights)

# create array of state background frequencies
# Takes responsibility of setModelRateMatrix() -> setBeagleStateFrequencies()
freqs = bg.createPatternWeights([0.25]*4)
bg.beagleSetStateFrequencies(instance, 0, freqs)

# create an array containing site category weights and rates
# Takes responsibility of setModelRateMatrix() and setDiscreteGammaShape() ->
# setBeagleAmongSiteRateVariationRates and setBeagleAmongSiteRateVariationProbs
weights = bg.createPatternWeights([1.0])
rates = bg.createPatternWeights([1.0])
bg.beagleSetCategoryWeights(instance, 0, weights)
bg.beagleSetCategoryRates(instance, rates)

# an eigen decomposition for the JC69 model
# Takes responsibility of setModelRateMatrix() -> setBeagleEigenDecomposition()
eigvec = bg.createPatternWeights([1.0, 2.0, 0.0, 0.5,
                                  1.0, -2.0, 0.5, 0.0,
                                  1.0, 2.0, 0.0, -0.5,
                                  1.0, -2.0, -0.5, 0.0])
invvec = bg.createPatternWeights([0.25, 0.25, 0.25, 0.25,
                                  0.125, -0.125, 0.125, -0.125,
                                  0.0, 1.0, 0.0, -1.0,
                                  1.0, 0.0, -1.0, 0.0])
eigval = bg.createPatternWeights([0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333])

# set the Eigen decomposition
bg.beagleSetEigenDecomposition(instance, 0, eigvec, invvec, eigval)

# a list of indices and edge lengths
# these get used to tell beagle which edge length goes with which node

# create a list of partial likelihood update operations
# the order is [dest, sourceScaling, destScaling, source1, matrix1, source2, matrix2]
node_list = []
derv_list = []
edge_list = []
operations = bg.new_BeagleOperationArray(n_internals+n_edges)

op_index = 0
for node in tree.traverse("postorder"):
    if not node.is_root():
        node_list.append(node.id)
        derv_list.append(node.id + n_edges)  # derivative indices
        edge_list.append(node.dist)

    if not node.is_leaf():
        children = node.get_children()
        if outgroup in children:
            children.remove(outgroup)
        left_child, right_child = children
        op_list = [node.id, bg.BEAGLE_OP_NONE, bg.BEAGLE_OP_NONE,
                   left_child.id, left_child.id,
                   right_child.id, right_child.id]
        op = bg.make_operation(op_list)
        bg.BeagleOperationArray_setitem(operations, op_index, op)
        op_index += 1

nodeIndices = bg.make_intarray(node_list)
dervIndices = bg.make_intarray(derv_list)
edgeLengths = bg.make_doublearray(edge_list)

for node in tree.traverse("preorder"):
    if not node.is_root():
        parent = node.up
        if not parent.is_root():
            sibling = node.get_sisters()[0]
            op_list = [node.id + n_edges, bg.BEAGLE_OP_NONE, bg.BEAGLE_OP_NONE,
                       parent.id + n_edges, parent.id,
                       sibling.id, sibling.id]
            op = bg.make_operation(op_list)
            bg.BeagleOperationArray_setitem(operations, op_index, op)
            op_index += 1
        else:
            children = parent.get_children()
            children.remove(node)
            op_list = [node.id + n_edges, bg.BEAGLE_OP_NONE, bg.BEAGLE_OP_NONE,
                       children[0].id, children[0].id,
                       children[1].id, children[1].id]
            op = bg.make_operation(op_list)
            bg.BeagleOperationArray_setitem(operations, op_index, op)
            op_index += 1

# tell BEAGLE to populate the transition matrices for the above edge lengths
# Takes responsibility of updateTransitionMatrices()
bg.beagleUpdateTransitionMatrices(instance,  # instance
                                  0,  # eigenIndex
                                  nodeIndices,  # probabilityIndices
                                  # None,  # firstDerivativeIndices
                                  dervIndices,  # firstDerivativeIndices
                                  None,  # secondDerivativeIndices
                                  edgeLengths,  # edgeLengths
                                  len(node_list))  # count

# this invokes all the math to carry out the likelihood calculation
cumulative_scale_index = bg.BEAGLE_OP_NONE
# cumulative_scale_index = 0
bg.beagleUpdatePartials(instance,  # instance
                        operations,  # eigenIndex
                        n_internals+n_edges,  # operationCount
                        cumulative_scale_index)  # cumulative scale index

categoryWeightIndex = bg.make_intarray([0])
stateFrequencyIndex = bg.make_intarray([0])
cumulativeScaleIndex = bg.make_intarray([cumulative_scale_index])

for node in tree.traverse('preorder'):
    if not node.is_root():
        upper_partials_index = bg.make_intarray([node.id + n_edges])
        node_index = bg.make_intarray([node.id])
        node_deriv_index = bg.make_intarray([node.id + n_edges])
        logLp = bg.new_doublep()
        dlogLp = bg.new_doublep()
        bg.beagleCalculateEdgeLogLikelihoods(
            instance,  # instance number
            upper_partials_index,  # indices of parent partialsBuffers
            node_index,  # indices of child partialsBuffers
            node_index,  # transition probability matrices for this edge
            # None,                   # first derivative matrices
            node_deriv_index,  # first derivative matrices
            None,  # second derivative matrices
            categoryWeightIndex,  # weights to apply to each partialsBuffer
            stateFrequencyIndex,  # state frequencies for each partialsBuffer
            cumulativeScaleIndex,  # scaleBuffers containing accumulated factors
            1,  # Number of partialsBuffer
            logLp,  # destination for log likelihood
            dlogLp,  # destination for first derivative  # derivative code
            None  # destination for second derivative
        )

        logL = bg.doublep_value(logLp)
        dlogL = bg.doublep_value(dlogLp)
        print(f'{node.id} logL: {logL} dlogL: {dlogL} {true_grad[node.name]} diff {dlogL-true_grad[node.name]}')

print("Woof!")
