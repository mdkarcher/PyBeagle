import beagle as bg
import phyloinfer as pinf
import numpy as np
import random as rnd

from models import *
from loglikelihood import *

ntips = 4
nsites = 10
true_tree = pinf.tree.create(ntips, branch=None)
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

# Beagle section

tree = true_tree.copy()
id_attr = "name"
leaf_attr = "name"

# Set up

table = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
         'a': 0, 'c': 1, 'g': 2, 't': 3,
         0: 0, 1: 1, 2: 2, 3: 3, '-': 4}

n_taxa, n_patterns = data.shape
n_states = 4

n_internals = n_taxa - 2
n_transition_probs = 2*n_taxa - 3


outgroup = next(tree.iter_search_nodes(name=0))

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
    n_internals,        # partials
    n_taxa,             # sequences
    n_states,           # states
    n_patterns,         # patterns
    1,                  # models
    n_transition_probs, # transition matrices
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
edge_list = []
operations = bg.new_BeagleOperationArray(n_internals)
op_index = 0
for node in reversed(list(tree.traverse("levelorder"))):
    if node is not outgroup:
        # print(f"Node is {getattr(node, id_attr)}")
        if node.is_root():
            # print(f"Adding outgroup {getattr(outgroup, id_attr)}")
            node_list.append(getattr(outgroup, id_attr))
            edge_list.append(outgroup.dist)
        else:
            # print(f"Adding node {getattr(node, id_attr)}")
            node_list.append(getattr(node, id_attr))
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
edgeLengths = bg.make_doublearray(edge_list)

# tell BEAGLE to populate the transition matrices for the above edge lengths
# Takes responsibility of updateTransitionMatrices()
bg.beagleUpdateTransitionMatrices(instance,  # instance
                                  0,  # eigenIndex
                                  nodeIndices,  # probabilityIndices
                                  None,  # firstDerivativeIndices
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

logLp = bg.new_doublep()
# rootIndex = bg.make_intarray([4])
categoryWeightIndex = bg.make_intarray([0])
stateFrequencyIndex = bg.make_intarray([0])
cumulativeScaleIndex = bg.make_intarray([cumulative_scale_index])

indexFocalParent = bg.make_intarray([getattr(tree, id_attr)])
indexFocalChild = bg.make_intarray([getattr(outgroup, id_attr)])

bg.beagleCalculateEdgeLogLikelihoods(
    instance,               # instance number
    indexFocalParent,       # indices of parent partialsBuffers
    indexFocalChild,        # indices of child partialsBuffers
    indexFocalChild,        # transition probability matrices for this edge
    None,                   # first derivative matrices
    None,                   # second derivative matrices
    categoryWeightIndex,    # weights to apply to each partialsBuffer
    stateFrequencyIndex,    # state frequencies for each partialsBuffer
    cumulativeScaleIndex,   # scaleBuffers containing accumulated factors
    1,                      # Number of partialsBuffer
    logLp,                  # destination for log likelihood
    None,                   # destination for first derivative
    None                    # destination for second derivative
)

logL = bg.doublep_value(logLp)
print(logL)
round(logL - true_llik, 6)
print("Woof!")


