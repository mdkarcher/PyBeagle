import beagle as bg
import phyloinfer as pinf
import numpy as np
import random as rnd

from models import *
from loglikelihood import *

ntips = 3
nsites = 10
true_tree = pinf.tree.create(ntips, branch="random")
print(true_tree)
true_tree.show()

data = pinf.data.treeSimu(true_tree, JC.D, JC.U, JC.U_inv, JC.pi, nsites)

L = pinf.Loglikelihood.initialCLV(data)
true_branch = pinf.branch.get(true_tree)

true_llik = pinf.Loglikelihood.phyloLoglikelihood(true_tree, true_branch, JC.D, JC.U, JC.U_inv, JC.pi, L)
print(f"The log-likelihood of the true tree: {true_llik}")

my_llik = loglikelihood(true_tree, data)
print(f"My log-likelihood of the true tree: {my_llik}")
round(my_llik - true_llik, 6)

true_grad = pinf.Loglikelihood.phyloLoglikelihood(true_tree, true_branch, JC.D, JC.U, JC.U_inv, JC.pi, L, grad=True)
np.round(true_grad, 6)

true_my_grad = gradient_loglikelihood(true_tree, data)
for i in range(len(true_my_grad)):
    print(f"Node {i}: ch={round(true_grad[i], 6)}, mk={round(true_my_grad[i], 6)}, diff={round(true_grad[i]-true_my_grad[i], 6)}")


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

n_partials = 2*n_internals
n_transition_probs = n_edges
n_derivatives = n_edges
n_matrices = n_transition_probs + n_derivatives


outgroup = next(tree.iter_search_nodes(name=0))
refresh_ids(tree, attr=id_attr)

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
operations = bg.new_BeagleOperationArray(n_internals)
op_index = 0
for node in reversed(list(tree.traverse("levelorder"))):
    if node is not outgroup:
        # print(f"Node is {getattr(node, id_attr)}")
        if node.is_root():
            # print(f"Adding outgroup {getattr(outgroup, id_attr)}")
            node_list.append(getattr(outgroup, id_attr))
            derv_list.append(getattr(outgroup, id_attr) + n_edges)  # derivative indices
            edge_list.append(outgroup.dist)
        else:
            # print(f"Adding node {getattr(node, id_attr)}")
            node_list.append(getattr(node, id_attr))
            derv_list.append(getattr(node, id_attr) + n_edges)  # derivative indices
            edge_list.append(node.dist)

        if not node.is_leaf():
            children = node.get_children()
            if outgroup in children:
                children.remove(outgroup)
            left_child, right_child = children
            op_list = [getattr(node, id_attr), bg.BEAGLE_OP_NONE, bg.BEAGLE_OP_NONE,
            # op_list = [getattr(node, id_attr), getattr(node, id_attr) - n_taxa + 1, bg.BEAGLE_OP_NONE,
            # op_list = [getattr(node, id_attr), op_index + 1, bg.BEAGLE_OP_NONE,
                       getattr(left_child, id_attr), getattr(left_child, id_attr),
                       getattr(right_child, id_attr), getattr(right_child, id_attr)]
            # print(f"Adding operation {op_list}")
            op = bg.make_operation(op_list)
            bg.BeagleOperationArray_setitem(operations, op_index, op)
            op_index += 1
nodeIndices = bg.make_intarray(node_list)
dervIndices = bg.make_intarray(derv_list)
edgeLengths = bg.make_doublearray(edge_list)

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
                        n_internals,  # operationCount
                        cumulative_scale_index)  # cumulative scale index

categoryWeightIndex = bg.make_intarray([0])
stateFrequencyIndex = bg.make_intarray([0])
cumulativeScaleIndex = bg.make_intarray([cumulative_scale_index])

indexFocalParent = bg.make_intarray([getattr(tree, id_attr)])

# Likelihood and derivative for outgroup/node 0

indexFocalChild0 = bg.make_intarray([getattr(outgroup, id_attr)])
indexFocalChildDerv0 = bg.make_intarray([getattr(outgroup, id_attr) + n_edges])

logL0p = bg.new_doublep()
derv0p = bg.new_doublep()

bg.beagleCalculateEdgeLogLikelihoods(
    instance,               # instance number
    indexFocalParent,       # indices of parent partialsBuffers
    indexFocalChild0,        # indices of child partialsBuffers
    indexFocalChild0,        # transition probability matrices for this edge
    # None,                   # first derivative matrices
    indexFocalChildDerv0,   # first derivative matrices
    None,                   # second derivative matrices
    categoryWeightIndex,    # weights to apply to each partialsBuffer
    stateFrequencyIndex,    # state frequencies for each partialsBuffer
    cumulativeScaleIndex,   # scaleBuffers containing accumulated factors
    1,                      # Number of partialsBuffer
    logL0p,                  # destination for log likelihood
    derv0p,                   # destination for first derivative  # derivative code
    None                    # destination for second derivative
)

logL0 = bg.doublep_value(logL0p)
print(logL0)
round(logL0 - true_llik, 6)

derv0 = bg.doublep_value(derv0p)
print(derv0)
round(derv0 - true_grad[0], 6)

# Likelihood? and derivative for node 1

node1 = next(tree.iter_search_nodes(**{id_attr: 1}))
indexFocalChild1 = bg.make_intarray([getattr(node1, id_attr)])
indexFocalChildDerv1 = bg.make_intarray([getattr(node1, id_attr) + n_edges])

logL1p = bg.new_doublep()
derv1p = bg.new_doublep()

bg.beagleCalculateEdgeLogLikelihoods(
    instance,               # instance number
    indexFocalParent,       # indices of parent partialsBuffers
    indexFocalChild1,        # indices of child partialsBuffers
    indexFocalChild1,        # transition probability matrices for this edge
    # None,                   # first derivative matrices
    indexFocalChildDerv1,   # first derivative matrices
    None,                   # second derivative matrices
    categoryWeightIndex,    # weights to apply to each partialsBuffer
    stateFrequencyIndex,    # state frequencies for each partialsBuffer
    cumulativeScaleIndex,   # scaleBuffers containing accumulated factors
    1,                      # Number of partialsBuffer
    logL1p,                  # destination for log likelihood
    derv1p,                   # destination for first derivative  # derivative code
    None                    # destination for second derivative
)

logL1 = bg.doublep_value(logL1p)
print(logL1)
round(logL1 - true_llik, 6)

derv1 = bg.doublep_value(derv1p)
print(derv1)
round(derv1 - true_grad[1], 6)

# Likelihood? and derivative for node 2

node2 = next(tree.iter_search_nodes(**{id_attr: 2}))
indexFocalChild2 = bg.make_intarray([getattr(node2, id_attr)])
indexFocalChildDerv2 = bg.make_intarray([getattr(node2, id_attr) + n_edges])

logL2p = bg.new_doublep()
derv2p = bg.new_doublep()

bg.beagleCalculateEdgeLogLikelihoods(
    instance,               # instance number
    indexFocalParent,       # indices of parent partialsBuffers
    indexFocalChild2,        # indices of child partialsBuffers
    indexFocalChild2,        # transition probability matrices for this edge
    # None,                   # first derivative matrices
    indexFocalChildDerv2,   # first derivative matrices
    None,                   # second derivative matrices
    categoryWeightIndex,    # weights to apply to each partialsBuffer
    stateFrequencyIndex,    # state frequencies for each partialsBuffer
    cumulativeScaleIndex,   # scaleBuffers containing accumulated factors
    1,                      # Number of partialsBuffer
    logL2p,                  # destination for log likelihood
    derv2p,                   # destination for first derivative  # derivative code
    None                    # destination for second derivative
)

logL2 = bg.doublep_value(logL2p)
print(logL2)
round(logL2 - true_llik, 6)

derv2 = bg.doublep_value(derv2p)
print(derv2)
round(derv2 - true_grad[2], 6)


print("Woof!")


