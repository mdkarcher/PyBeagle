# from collections import namedtuple
import numpy as np
# from ete3 import Tree
import beagle as bg

from models import *
from tree_utils import *
from sequence_data import *
from reroot import *


def loglikelihood(tree, seqs, model=JC, id_attr="id", leaf_attr="name"):
    """Calculate log-likelihood of a tree and sequences.

    :param tree: Ete3 Tree object representing the tree topology and branch lengths.
    :param seqs: Array-like or dictionary of lists of sequence characters or integers.
    :param model: Substitution model in namedtuple "Model". Default: Jukes-Cantor.
    :param id_attr: Attribute/feature of each node that uniquely identifies it.
    :param leaf_attr: Attribute/feature of each leaf that uniquely identifies it,
    and uniquely identifies the sequence data row/entry in seqs.
    :return: float representing the log-likelihood of the tree, given the sequence data and model.
    """

    tree = tree.copy()

    needs_refresh = False
    id_set = set()
    for node in tree.traverse():
        if id_attr not in node.features:
            needs_refresh = True
            break
        id_set.add(getattr(node, id_attr))
    if needs_refresh or len(id_set) < 2*len(tree)-2:
        refresh_ids(tree, id_attr)

    leaves = tree.get_leaves()
    assert len({getattr(leaf, leaf_attr) for leaf in leaves}) == len(leaves)

    seq_mats = convert_to_dict_mats(seqs)
    val = next(iter(seq_mats.values()))
    n_sites = val.shape[1]

    llik = 0
    down = dict() # [None] * (2*n_taxa - 3)
    Pt = dict() # [None] * (2*n_taxa - 3)
    partials = dict()
    for node in tree.traverse("postorder"):
        if not node.is_leaf():
            partials[getattr(node, id_attr)] = np.ones((4, n_sites))
            for child in node.children:
                # Pt[getattr(child, attr)] = model.U @ np.diag(np.exp(model.D * child.dist)) @ model.U_inv
                Pt[getattr(child, id_attr)] = prob_trans(child.dist, model)
                if child.is_leaf():
                    down[getattr(child, id_attr)] = Pt[getattr(child, id_attr)] @ seq_mats[getattr(child, leaf_attr)]
                else:
                    down[getattr(child, id_attr)] = Pt[getattr(child, id_attr)] @ partials[getattr(child, id_attr)]
                partials[getattr(node, id_attr)] *= down[getattr(child, id_attr)]
            scalar = np.amax(partials[getattr(node, id_attr)], axis=0)
            assert all(scalar)
            partials[getattr(node, id_attr)] /= scalar
            llik += np.sum(np.log(scalar))
    assert node == tree
    llik += np.sum(np.log(model.pi @ partials[getattr(tree, id_attr)]))
    return llik


def loglikelihood_beagle(tree, seqs, model=JC, id_attr=None, leaf_attr=None, scaling=False):
    if id_attr is None:
        id_attr = "id"
    if leaf_attr is None:
        leaf_attr = "name"

    tree = tree.copy()
    seq_dict = convert_to_dict_lists(seqs)
    val = next(iter(seq_dict.values()))
    n_taxa = len(seq_dict)
    n_patterns = len(val)
    n_states = len(model.pi)
    n_internals = n_taxa - 2
    n_transition_probs = 2 * n_taxa - 3

    n_scale_buffers = 0
    if scaling:
        n_scale_buffers = n_internals + 1

    outgroup = next(iter(tree.get_leaves()))
    reroot(tree, outgroup)
    refresh_ids(tree, attr=id_attr)

    # Sanity check(s)

    assert set(getattr(node, id_attr) for node in tree.get_leaves()) == set(range(len(tree)))
    assert set(getattr(node, id_attr) for node in tree.traverse()) == set(range(2 * len(tree) - 2))

    # Instantiate Beagle

    requirementFlags = 0

    if scaling:
        requirementFlags |= bg.BEAGLE_FLAG_SCALING_MANUAL

    returnInfo = bg.BeagleInstanceDetails()
    instance = bg.beagleCreateInstance(
        n_taxa,  # tips
        n_internals,  # partials
        n_taxa,  # sequences
        n_states,  # states
        n_patterns,  # patterns
        1,  # models
        n_transition_probs,  # transition matrices
        1,  # rate categories
        n_scale_buffers,  # scale buffers
        None,  # resource restrictions
        0,  # length of resource list
        0,  # preferred flags
        requirementFlags,  # required flags
        returnInfo  # output details
    )

    assert instance >= 0

    # Set tip states block
    for node in tree.get_leaves():
        states = bg.make_intarray(seq_dict[getattr(node, leaf_attr)])
        # states = bg.createStates(seqs[getattr(node, leaf_attr)], dna_ids)
        bg.beagleSetTipStates(instance, getattr(node, id_attr), states)

    patternWeights = bg.createPatternWeights([1] * n_patterns)
    bg.beagleSetPatternWeights(instance, patternWeights)

    # create array of state background frequencies
    freqs = bg.createPatternWeights(model.pi)
    # freqs = bg.createPatternWeights([0.25] * 4)
    bg.beagleSetStateFrequencies(instance, 0, freqs)

    # create an array containing site category weights and rates
    weights = bg.createPatternWeights([1.0])
    rates = bg.createPatternWeights([1.0])
    bg.beagleSetCategoryWeights(instance, 0, weights)
    bg.beagleSetCategoryRates(instance, rates)

    # set the Eigen decomposition
    eigvec = bg.createPatternWeights(model.U.ravel())
    invvec = bg.createPatternWeights(model.U_inv.ravel())
    eigval = bg.createPatternWeights(model.D)
    # eigvec = bg.createPatternWeights([1.0, 2.0, 0.0, 0.5,
    #                                   1.0, -2.0, 0.5, 0.0,
    #                                   1.0, 2.0, 0.0, -0.5,
    #                                   1.0, -2.0, -0.5, 0.0])
    # invvec = bg.createPatternWeights([0.25, 0.25, 0.25, 0.25,
    #                                   0.125, -0.125, 0.125, -0.125,
    #                                   0.0, 1.0, 0.0, -1.0,
    #                                   1.0, 0.0, -1.0, 0.0])
    # eigval = bg.createPatternWeights([0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333])

    bg.beagleSetEigenDecomposition(instance, 0, eigvec, invvec, eigval)

    # a list of indices and edge lengths
    # create a list of partial likelihood update operations
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

                scaling_index = bg.BEAGLE_OP_NONE
                if scaling:
                    scaling_index = op_index + 1

                op_list = [getattr(node, id_attr), scaling_index, bg.BEAGLE_OP_NONE,
                           getattr(left_child, id_attr), getattr(left_child, id_attr),
                           getattr(right_child, id_attr), getattr(right_child, id_attr)]
                # print(f"Adding operation {op_list}")
                op = bg.make_operation(op_list)
                bg.BeagleOperationArray_setitem(operations, op_index, op)
                op_index += 1
    nodeIndices = bg.make_intarray(node_list)
    edgeLengths = bg.make_doublearray(edge_list)

    # tell BEAGLE to populate the transition matrices for the above edge lengths
    bg.beagleUpdateTransitionMatrices(instance,  # instance
                                      0,  # eigenIndex
                                      nodeIndices,  # probabilityIndices
                                      None,  # firstDerivativeIndices
                                      None,  # secondDerivativeIndices
                                      edgeLengths,  # edgeLengths
                                      len(node_list))  # count

    # this invokes all the math to carry out the likelihood calculation
    cumulative_scale_index = bg.BEAGLE_OP_NONE
    if scaling:
        cumulative_scale_index = 0
        bg.beagleResetScaleFactors(instance, cumulative_scale_index)
    bg.beagleUpdatePartials(instance,  # instance
                            operations,  # eigenIndex
                            n_internals,  # operationCount
                            cumulative_scale_index)  # cumulative scale index

    logLp = bg.new_doublep()
    categoryWeightIndex = bg.make_intarray([0])
    stateFrequencyIndex = bg.make_intarray([0])
    cumulativeScaleIndex = bg.make_intarray([cumulative_scale_index])

    indexFocalParent = bg.make_intarray([getattr(tree, id_attr)])
    indexFocalChild = bg.make_intarray([getattr(outgroup, id_attr)])

    bg.beagleCalculateEdgeLogLikelihoods(
        instance,  # instance number
        indexFocalParent,  # indices of parent partialsBuffers
        indexFocalChild,  # indices of child partialsBuffers
        indexFocalChild,  # transition probability matrices for this edge
        None,  # first derivative matrices
        None,  # second derivative matrices
        categoryWeightIndex,  # weights to apply to each partialsBuffer
        stateFrequencyIndex,  # state frequencies for each partialsBuffer
        cumulativeScaleIndex,  # scaleBuffers containing accumulated factors
        1,  # Number of partialsBuffer
        logLp,  # destination for log likelihood
        None,  # destination for first derivative
        None  # destination for second derivative
    )

    logL = bg.doublep_value(logLp)
    return logL


def loglikelihood_beagle_rooting_test(tree, seqs, model=JC, id_attr="id", leaf_attr="name", scaling=False):
    # if id_attr is None:
    #     id_attr = "id"
    # if leaf_attr is None:
    #     leaf_attr = "name"

    tree = tree.copy()
    seq_dict = convert_to_dict_lists(seqs)
    val = next(iter(seq_dict.values()))
    n_taxa = len(seq_dict)
    n_patterns = len(val)
    n_states = len(model.pi)
    n_internals = n_taxa - 2
    n_transition_probs = 2 * n_taxa - 3

    n_scale_buffers = 0
    if scaling:
        n_scale_buffers = n_internals + 1

    # outgroup = next(iter(tree.get_leaves()))
    # reroot(tree, outgroup)
    outgroup = tree.children[0]
    refresh_ids(tree, attr=id_attr)

    # Sanity check(s)

    assert set(getattr(node, id_attr) for node in tree.get_leaves()) == set(range(len(tree)))
    assert set(getattr(node, id_attr) for node in tree.traverse()) == set(range(2 * len(tree) - 2))

    # Instantiate Beagle

    requirementFlags = 0

    if scaling:
        requirementFlags |= bg.BEAGLE_FLAG_SCALING_MANUAL

    returnInfo = bg.BeagleInstanceDetails()
    instance = bg.beagleCreateInstance(
        n_taxa,  # tips
        n_internals,  # partials
        n_taxa,  # sequences
        n_states,  # states
        n_patterns,  # patterns
        1,  # models
        n_transition_probs,  # transition matrices
        1,  # rate categories
        n_scale_buffers,  # scale buffers
        None,  # resource restrictions
        0,  # length of resource list
        0,  # preferred flags
        requirementFlags,  # required flags
        returnInfo  # output details
    )

    assert instance >= 0

    # Set tip states block
    for node in tree.get_leaves():
        states = bg.make_intarray(seq_dict[getattr(node, leaf_attr)])
        # states = bg.createStates(seqs[getattr(node, leaf_attr)], dna_ids)
        bg.beagleSetTipStates(instance, getattr(node, id_attr), states)

    patternWeights = bg.createPatternWeights([1] * n_patterns)
    bg.beagleSetPatternWeights(instance, patternWeights)

    # create array of state background frequencies
    freqs = bg.createPatternWeights(model.pi)
    # freqs = bg.createPatternWeights([0.25] * 4)
    bg.beagleSetStateFrequencies(instance, 0, freqs)

    # create an array containing site category weights and rates
    weights = bg.createPatternWeights([1.0])
    rates = bg.createPatternWeights([1.0])
    bg.beagleSetCategoryWeights(instance, 0, weights)
    bg.beagleSetCategoryRates(instance, rates)

    # set the Eigen decomposition
    eigvec = bg.createPatternWeights(model.U.ravel())
    invvec = bg.createPatternWeights(model.U_inv.ravel())
    eigval = bg.createPatternWeights(model.D)
    # eigvec = bg.createPatternWeights([1.0, 2.0, 0.0, 0.5,
    #                                   1.0, -2.0, 0.5, 0.0,
    #                                   1.0, 2.0, 0.0, -0.5,
    #                                   1.0, -2.0, -0.5, 0.0])
    # invvec = bg.createPatternWeights([0.25, 0.25, 0.25, 0.25,
    #                                   0.125, -0.125, 0.125, -0.125,
    #                                   0.0, 1.0, 0.0, -1.0,
    #                                   1.0, 0.0, -1.0, 0.0])
    # eigval = bg.createPatternWeights([0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333])

    bg.beagleSetEigenDecomposition(instance, 0, eigvec, invvec, eigval)

    # a list of indices and edge lengths
    # create a list of partial likelihood update operations
    node_list = []
    edge_list = []
    operations = bg.new_BeagleOperationArray(n_internals)
    op_index = 0
    for node in reversed(list(tree.traverse("levelorder"))):
        if not node.is_root():
            node_list.append(getattr(node, id_attr))
            edge_list.append(node.dist)

        if not node.is_leaf():
            children = node.get_children()
            if outgroup in children:
                children.remove(outgroup)
            left_child, right_child = children

            scaling_index = bg.BEAGLE_OP_NONE
            if scaling:
                scaling_index = op_index + 1

            op_list = [getattr(node, id_attr), scaling_index, bg.BEAGLE_OP_NONE,
                       getattr(left_child, id_attr), getattr(left_child, id_attr),
                       getattr(right_child, id_attr), getattr(right_child, id_attr)]
            op = bg.make_operation(op_list)
            bg.BeagleOperationArray_setitem(operations, op_index, op)
            op_index += 1
    nodeIndices = bg.make_intarray(node_list)
    edgeLengths = bg.make_doublearray(edge_list)

    # tell BEAGLE to populate the transition matrices for the above edge lengths
    bg.beagleUpdateTransitionMatrices(instance,  # instance
                                      0,  # eigenIndex
                                      nodeIndices,  # probabilityIndices
                                      None,  # firstDerivativeIndices
                                      None,  # secondDerivativeIndices
                                      edgeLengths,  # edgeLengths
                                      len(node_list))  # count

    # this invokes all the math to carry out the likelihood calculation
    cumulative_scale_index = bg.BEAGLE_OP_NONE
    if scaling:
        cumulative_scale_index = 0
        bg.beagleResetScaleFactors(instance, cumulative_scale_index)
    bg.beagleUpdatePartials(instance,  # instance
                            operations,  # eigenIndex
                            n_internals,  # operationCount
                            cumulative_scale_index)  # cumulative scale index

    logLp = bg.new_doublep()
    categoryWeightIndex = bg.make_intarray([0])
    stateFrequencyIndex = bg.make_intarray([0])
    cumulativeScaleIndex = bg.make_intarray([cumulative_scale_index])

    indexFocalParent = bg.make_intarray([getattr(tree, id_attr)])
    indexFocalChild = bg.make_intarray([getattr(outgroup, id_attr)])

    bg.beagleCalculateEdgeLogLikelihoods(
        instance,  # instance number
        indexFocalParent,  # indices of parent partialsBuffers
        indexFocalChild,  # indices of child partialsBuffers
        indexFocalChild,  # transition probability matrices for this edge
        None,  # first derivative matrices
        None,  # second derivative matrices
        categoryWeightIndex,  # weights to apply to each partialsBuffer
        stateFrequencyIndex,  # state frequencies for each partialsBuffer
        cumulativeScaleIndex,  # scaleBuffers containing accumulated factors
        1,  # Number of partialsBuffer
        logLp,  # destination for log likelihood
        None,  # destination for first derivative
        None  # destination for second derivative
    )

    logL = bg.doublep_value(logLp)
    return logL


def loglikelihood_beagle_init(seqs, model=JC, scaling=False):
    # if id_attr is None:
    #     id_attr = "id"
    # if leaf_attr is None:
    #     leaf_attr = "name"

    # tree = tree.copy()
    seq_dict = convert_to_dict_lists(seqs)
    val = next(iter(seq_dict.values()))
    n_taxa = len(seq_dict)
    n_patterns = len(val)
    n_states = len(model.pi)
    n_internals = n_taxa - 2
    n_transition_probs = 2 * n_taxa - 3

    n_scale_buffers = 0
    if scaling:
        n_scale_buffers = n_internals + 1

    # outgroup = next(iter(tree.get_leaves()))
    # reroot(tree, outgroup)
    # outgroup = tree.children[0]
    # refresh_ids(tree, attr=id_attr)

    # Sanity check(s)

    # assert set(getattr(node, id_attr) for node in tree.get_leaves()) == set(range(len(tree)))
    # assert set(getattr(node, id_attr) for node in tree.traverse()) == set(range(2 * len(tree) - 2))

    # Instantiate Beagle

    requirementFlags = 0

    if scaling:
        requirementFlags |= bg.BEAGLE_FLAG_SCALING_MANUAL

    returnInfo = bg.BeagleInstanceDetails()
    instance = bg.beagleCreateInstance(
        n_taxa,  # tips
        n_internals,  # partials
        n_taxa,  # sequences
        n_states,  # states
        n_patterns,  # patterns
        1,  # models
        n_transition_probs,  # transition matrices
        1,  # rate categories
        n_scale_buffers,  # scale buffers
        None,  # resource restrictions
        0,  # length of resource list
        0,  # preferred flags
        requirementFlags,  # required flags
        returnInfo  # output details
    )

    assert instance >= 0

    # Set tip states block
    # for node in tree.get_leaves():
    tip_name_to_address = {}
    for i, key in enumerate(seq_dict):
        states = bg.make_intarray(seq_dict[key])
        # states = bg.createStates(seqs[getattr(node, leaf_attr)], dna_ids)
        bg.beagleSetTipStates(instance, i, states)
        tip_name_to_address[key] = i

    patternWeights = bg.createPatternWeights([1] * n_patterns)
    bg.beagleSetPatternWeights(instance, patternWeights)

    # create array of state background frequencies
    freqs = bg.createPatternWeights(model.pi)
    # freqs = bg.createPatternWeights([0.25] * 4)
    bg.beagleSetStateFrequencies(instance, 0, freqs)

    # create an array containing site category weights and rates
    weights = bg.createPatternWeights([1.0])
    rates = bg.createPatternWeights([1.0])
    bg.beagleSetCategoryWeights(instance, 0, weights)
    bg.beagleSetCategoryRates(instance, rates)

    # set the Eigen decomposition
    eigvec = bg.createPatternWeights(model.U.ravel())
    invvec = bg.createPatternWeights(model.U_inv.ravel())
    eigval = bg.createPatternWeights(model.D)
    # eigvec = bg.createPatternWeights([1.0, 2.0, 0.0, 0.5,
    #                                   1.0, -2.0, 0.5, 0.0,
    #                                   1.0, 2.0, 0.0, -0.5,
    #                                   1.0, -2.0, -0.5, 0.0])
    # invvec = bg.createPatternWeights([0.25, 0.25, 0.25, 0.25,
    #                                   0.125, -0.125, 0.125, -0.125,
    #                                   0.0, 1.0, 0.0, -1.0,
    #                                   1.0, 0.0, -1.0, 0.0])
    # eigval = bg.createPatternWeights([0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333])

    bg.beagleSetEigenDecomposition(instance, 0, eigvec, invvec, eigval)
    return instance, tip_name_to_address


def loglikelihood_beagle_evaluate(instance, tree, tip_name_to_address, id_attr="id", leaf_attr="name", scaling=False):
    n_taxa = len(tree)
    # n_patterns = len(val)
    # n_states = len(model.pi)
    n_internals = n_taxa - 2
    n_transition_probs = 2 * n_taxa - 3

    n_scale_buffers = 0
    if scaling:
        n_scale_buffers = n_internals + 1

    outgroup = tree.children[0]
    refresh_ids(tree, attr=id_attr)

    # Sanity check(s)

    assert set(getattr(node, id_attr) for node in tree.get_leaves()) == set(range(len(tree)))
    assert set(getattr(node, id_attr) for node in tree.traverse()) == set(range(2 * len(tree) - 2))

    # a list of indices and edge lengths
    # create a list of partial likelihood update operations
    node_list = []
    edge_list = []
    operations = bg.new_BeagleOperationArray(n_internals)
    op_index = 0
    for node in reversed(list(tree.traverse("levelorder"))):
        if not node.is_root():
            node_list.append(getattr(node, id_attr))
            edge_list.append(node.dist)

        if not node.is_leaf():
            children = node.get_children()
            if outgroup in children:
                children.remove(outgroup)
            left_child, right_child = children

            scaling_index = bg.BEAGLE_OP_NONE
            if scaling:
                scaling_index = op_index + 1

            node_address = getattr(node, id_attr)
            if left_child.is_leaf():
                left_child_address = tip_name_to_address[getattr(left_child, leaf_attr)]
            else:
                left_child_address = getattr(left_child, id_attr)
            if right_child.is_leaf():
                right_child_address = tip_name_to_address[getattr(right_child, leaf_attr)]
            else:
                right_child_address = getattr(right_child, id_attr)
            op_list = [node_address, scaling_index, bg.BEAGLE_OP_NONE,
                       left_child_address, left_child_address,
                       right_child_address, right_child_address]
            op = bg.make_operation(op_list)
            bg.BeagleOperationArray_setitem(operations, op_index, op)
            op_index += 1
    nodeIndices = bg.make_intarray(node_list)
    edgeLengths = bg.make_doublearray(edge_list)

    # tell BEAGLE to populate the transition matrices for the above edge lengths
    bg.beagleUpdateTransitionMatrices(instance,  # instance
                                      0,  # eigenIndex
                                      nodeIndices,  # probabilityIndices
                                      None,  # firstDerivativeIndices
                                      None,  # secondDerivativeIndices
                                      edgeLengths,  # edgeLengths
                                      len(node_list))  # count

    # this invokes all the math to carry out the likelihood calculation
    cumulative_scale_index = bg.BEAGLE_OP_NONE
    if scaling:
        cumulative_scale_index = 0
        bg.beagleResetScaleFactors(instance, cumulative_scale_index)
    bg.beagleUpdatePartials(instance,  # instance
                            operations,  # eigenIndex
                            n_internals,  # operationCount
                            cumulative_scale_index)  # cumulative scale index

    logLp = bg.new_doublep()
    categoryWeightIndex = bg.make_intarray([0])
    stateFrequencyIndex = bg.make_intarray([0])
    cumulativeScaleIndex = bg.make_intarray([cumulative_scale_index])

    indexFocalParent = bg.make_intarray([getattr(tree, id_attr)])
    indexFocalChild = bg.make_intarray([getattr(outgroup, id_attr)])

    bg.beagleCalculateEdgeLogLikelihoods(
        instance,  # instance number
        indexFocalParent,  # indices of parent partialsBuffers
        indexFocalChild,  # indices of child partialsBuffers
        indexFocalChild,  # transition probability matrices for this edge
        None,  # first derivative matrices
        None,  # second derivative matrices
        categoryWeightIndex,  # weights to apply to each partialsBuffer
        stateFrequencyIndex,  # state frequencies for each partialsBuffer
        cumulativeScaleIndex,  # scaleBuffers containing accumulated factors
        1,  # Number of partialsBuffer
        logLp,  # destination for log likelihood
        None,  # destination for first derivative
        None  # destination for second derivative
    )

    logL = bg.doublep_value(logLp)
    return logL

