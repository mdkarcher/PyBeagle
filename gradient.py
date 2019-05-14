# from collections import namedtuple
import numpy as np
# from ete3 import Tree
import beagle as bg

from models import *
from tree_utils import *
from sequence_data import *
from reroot import *


def gradient_loglikelihood(tree, seqs, model=JC, id_attr="id", leaf_attr="name"):
    """Calculate branch length gradient of a tree and sequences.

    :param tree: Ete3 Tree object representing the tree topology and branch lengths.
    :param seqs: Array-like or dictionary of lists of sequence characters or integers.
    :param model: Substitution model in namedtuple "Model". Default: Jukes-Cantor.
    :param id_attr: Attribute/feature of each node that uniquely identifies it.
    :param leaf_attr: Attribute/feature of each leaf that uniquely identifies it,
    and uniquely identifies the sequence data row/entry in seqs.
    :return: float representing the derivative of the log-likelihood of the tree,
    given the sequence data and model.
    """

    tree = tree.copy()

    needs_refresh = False
    id_set = set()
    for node in tree.traverse():
        if id_attr not in node.features:
            needs_refresh = True
            break
        id_set.add(getattr(node, id_attr))
    if needs_refresh or len(id_set) < 2 * len(tree) - 2:
        refresh_ids(tree, id_attr)

    leaves = tree.get_leaves()
    assert len({getattr(leaf, leaf_attr) for leaf in leaves}) == len(leaves)

    seq_mats = convert_to_dict_mats(seqs)
    val = next(iter(seq_mats.values()))
    n_sites = val.shape[1]
    n_states = len(model.pi)
    ones_mat = np.ones((n_states, n_sites))

    llik = 0
    down = dict()
    Pt = dict()
    partials = dict()
    for node in tree.traverse("postorder"):
        if not node.is_leaf():
            partials[getattr(node, id_attr)] = ones_mat.copy()
            for child in node.children:
                Pt[getattr(child, id_attr)] = prob_trans(child.dist, model)
                down[getattr(child, id_attr)] = Pt[getattr(child, id_attr)] @ partials[getattr(child, id_attr)]
                partials[getattr(node, id_attr)] *= down[getattr(child, id_attr)]
            scalar = np.amax(partials[getattr(node, id_attr)], axis=0)
            assert all(scalar)
            partials[getattr(node, id_attr)] /= scalar
            # llik += np.sum(np.log(scalar))
        else:
            partials[getattr(node, id_attr)] = seq_mats[getattr(node, leaf_attr)]
    # llik += np.sum(np.log(model.pi @ partials[getattr(tree, id_attr)]))
    # return llik

    up = dict()
    grad = dict()
    for node in tree.traverse("preorder"):
        if node.is_root():
            up[getattr(node, id_attr)] = (model.pi * ones_mat.T).T
        else:
            up[getattr(node, id_attr)] = ones_mat.copy()
            for sister in node.get_sisters():
                up[getattr(node, id_attr)] *= down[getattr(sister, id_attr)]
            up[getattr(node, id_attr)] *= up[getattr(node.up, id_attr)]
            # pt_matrix_grad = np.dot(U, np.dot(np.diag(D * np.exp(D * branch[node.name])), U_inv))
            pt_matrix_grad = grad_trans(node.dist, model)

            gradient = np.sum(up[getattr(node, id_attr)] * (pt_matrix_grad @ partials[getattr(node, id_attr)]), axis=0)

            up[getattr(node, id_attr)] = (Pt[getattr(node, id_attr)].T @ up[getattr(node, id_attr)])
            gradient /= np.sum(up[getattr(node, id_attr)] * partials[getattr(node, id_attr)], axis=0)
            grad[getattr(node, id_attr)] = np.sum(gradient)

            if not node.is_leaf():
                scalar = np.amax(up[getattr(node, id_attr)], axis=0)
                up[getattr(node, id_attr)] /= scalar
    return grad


def gradient_loglikelihood_beagle(tree, seqs, model=JC, id_attr=None, leaf_attr=None, scaling=False):
    """Calculate branch length gradient of a tree and sequences.

    :param tree: Ete3 Tree object representing the tree topology and branch lengths.
    :param seqs: Array-like or dictionary of lists of sequence characters or integers.
    :param model: Substitution model in namedtuple "Model". Default: Jukes-Cantor.
    :param id_attr: Attribute/feature of each node that uniquely identifies it.
    :param leaf_attr: Attribute/feature of each leaf that uniquely identifies it,
    and uniquely identifies the sequence data row/entry in seqs.
    :return: float representing the derivative of the log-likelihood of the tree,
    given the sequence data and model.
    """

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
    n_edges = 2 * n_taxa - 3

    n_partials = n_internals + n_edges
    n_transition_probs = n_edges
    n_derivatives = n_edges
    n_matrices = n_transition_probs + n_derivatives

    if scaling:
        print("Scaling not currently supported.")
        scaling = False
    n_scale_buffers = 0
    if scaling:
        n_scale_buffers = n_internals + 1

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
        n_partials,  # partials
        n_taxa,  # sequences
        n_states,  # states
        n_patterns,  # patterns
        1,  # models
        n_matrices,  # transition matrices
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
    derv_list = []
    edge_list = []
    operations = bg.new_BeagleOperationArray(n_internals + n_edges)

    op_index = 0
    for node in tree.traverse("postorder"):
        if not node.is_root():
            node_list.append(getattr(node, id_attr))
            derv_list.append(getattr(node, id_attr) + n_edges)  # derivative indices
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
    dervIndices = bg.make_intarray(derv_list)
    edgeLengths = bg.make_doublearray(edge_list)

    for node in tree.traverse("preorder"):
        if not node.is_root():
            parent = node.up
            if not parent.is_root():
                sibling = node.get_sisters()[0]
                op_list = [getattr(node, id_attr) + n_edges, bg.BEAGLE_OP_NONE, bg.BEAGLE_OP_NONE,
                           getattr(parent, id_attr) + n_edges, getattr(parent, id_attr),
                           getattr(sibling, id_attr), getattr(sibling, id_attr)]
                op = bg.make_operation(op_list)
                bg.BeagleOperationArray_setitem(operations, op_index, op)
                op_index += 1
            else:
                children = parent.get_children()
                children.remove(node)

                # TODO: Do I add scaling factors here?
                op_list = [getattr(node, id_attr) + n_edges, bg.BEAGLE_OP_NONE, bg.BEAGLE_OP_NONE,
                           getattr(children[0], id_attr), getattr(children[0], id_attr),
                           getattr(children[1], id_attr), getattr(children[1], id_attr)]
                op = bg.make_operation(op_list)
                bg.BeagleOperationArray_setitem(operations, op_index, op)
                op_index += 1

    # tell BEAGLE to populate the transition matrices for the above edge lengths
    bg.beagleUpdateTransitionMatrices(instance,  # instance
                                      0,  # eigenIndex
                                      nodeIndices,  # probabilityIndices
                                      dervIndices,  # firstDerivativeIndices
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
                            n_internals + n_edges,  # operationCount
                            cumulative_scale_index)  # cumulative scale index

    categoryWeightIndex = bg.make_intarray([0])
    stateFrequencyIndex = bg.make_intarray([0])
    cumulativeScaleIndex = bg.make_intarray([cumulative_scale_index])

    logLp = bg.new_doublep()
    dlogLp = bg.new_doublep()
    result = dict()

    for node in tree.traverse('preorder'):
        if not node.is_root():
            upper_partials_index = bg.make_intarray([getattr(node, id_attr) + n_edges])
            node_index = bg.make_intarray([getattr(node, id_attr)])
            node_deriv_index = bg.make_intarray([getattr(node, id_attr) + n_edges])
            bg.beagleCalculateEdgeLogLikelihoods(
                instance,  # instance number
                upper_partials_index,  # indices of parent partialsBuffers
                node_index,  # indices of child partialsBuffers
                node_index,  # transition probability matrices for this edge
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

            # logL = bg.doublep_value(logLp)
            dlogL = bg.doublep_value(dlogLp)
            result[getattr(node, id_attr)] = dlogL

    return result
