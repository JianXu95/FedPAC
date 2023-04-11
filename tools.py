import copy
import torch
import types
import math
import numpy as np
from scipy import stats
import torch.nn.functional as F
import cvxpy as cvx

def pairwise(data):
    """ Simple generator of the pairs (x, y) in a tuple such that index x < index y.
    Args:
    data Indexable (including ability to query length) containing the elements
    Returns:
    Generator over the pairs of the elements of 'data'
    """
    n = len(data)
    for i in range(n):
        for j in range(i, n):
            yield (data[i], data[j])

            
def get_protos(protos):
    """
    Returns the average of the feature embeddings of samples from per-class.
    """
    protos_mean = {}
    for [label, proto_list] in protos.items():
        proto = 0 * proto_list[0]
        for i in proto_list:
            proto += i
        protos_mean[label] = proto / len(proto_list)

    return protos_mean


def protos_aggregation(local_protos_list, local_sizes_list):
    agg_protos_label = {}
    agg_sizes_label = {}
    for idx in range(len(local_protos_list)):
        local_protos = local_protos_list[idx]
        local_sizes = local_sizes_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
                agg_sizes_label[label].append(local_sizes[label])
            else:
                agg_protos_label[label] = [local_protos[label]]
                agg_sizes_label[label] = [local_sizes[label]]

    for [label, proto_list] in agg_protos_label.items():
        sizes_list = agg_sizes_label[label]
        proto = 0 * proto_list[0]
        for i in range(len(proto_list)):
            proto += sizes_list[i] * proto_list[i]
        agg_protos_label[label] = proto / sum(sizes_list)

    return agg_protos_label


def average_weights_weighted(w, avg_weight):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    weight = torch.tensor(avg_weight)
    agg_w = weight/(weight.sum(dim=0))
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(len(w)):
            w_avg[key] += agg_w[i]*w[i][key]
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def agg_classifier_weighted_p(w, avg_weight, keys, idx):
    """
    Returns the average of the weights.
    """
    w_0 = copy.deepcopy(w[idx])
    for key in keys:
        w_0[key] = torch.zeros_like(w_0[key])
    wc = 0
    for i in range(len(w)):
        wi = avg_weight[i]
        wc += wi
        for key in keys:
            w_0[key] += wi*w[i][key]
    for key in keys:
        w_0[key] = torch.div(w_0[key], wc)
    return w_0

    
def get_head_agg_weight(num_users, Vars, Hs, *args, **kwargs):

    device = Hs[0][0].device
    num_cls = Hs[0].shape[0] # number of classes
    d = Hs[0].shape[1] # dimension of feature representation
    avg_weight = []
    for i in range(num_users):
        # ---------------------------------------------------------------------------
        # variance ter
        v = torch.tensor(Vars, device=device)
        # ---------------------------------------------------------------------------
        # bias term
        h_ref = Hs[i]
        dist = torch.zeros((num_users, num_users), device=device)
        for j1, j2 in pairwise(tuple(range(num_users))):
            h_j1 = Hs[j1]
            h_j2 = Hs[j2]
            h = torch.zeros((d, d), device=device)
            for k in range(num_cls):
                h += torch.mm((h_ref[k]-h_j1[k]).reshape(d,1), (h_ref[k]-h_j2[k]).reshape(1,d))
            dj12 = torch.trace(h)
            dist[j1][j2] = dj12
            dist[j2][j1] = dj12

        # QP solver
        p_matrix = torch.diag(v) + dist
        p_matrix = p_matrix.cpu().numpy()  # coefficient for QP problem
        evals, evecs = torch.eig(torch.tensor(p_matrix), eigenvectors=True)
        
        # for numerical stablity
        p_matrix_new = 0
        p_matrix_new = 0
        for ii in range(num_users):
            if evals[ii,0] >= 0.01:
                p_matrix_new += evals[ii,0]*torch.mm(evecs[:,ii].reshape(num_users,1), evecs[:,ii].reshape(1, num_users))
        p_matrix = p_matrix_new.numpy() if not np.all(np.linalg.eigvals(p_matrix)>=0.0) else p_matrix
        
        # solve QP
        alpha = 0
        eps = 1e-3
        if np.all(np.linalg.eigvals(p_matrix)>=0):
            alphav = cvx.Variable(num_users)
            obj = cvx.Minimize(cvx.quad_form(alphav, p_matrix))
            prob = cvx.Problem(obj, [cvx.sum(alphav) == 1.0, alphav >= 0])
            prob.solve()
            alpha = alphav.value
            alpha = [(i)*(i>eps) for i in alpha] # zero-out small weights (<eps)
            if i == 0:
                print('({}) Agg Weights of Classifier Head'.format(i+1))
                print(alpha,'\n')
            
        else:
            alpha = None # if no solution for the optimization problem, use local classifier only
        
        avg_weight.append(alpha)

    return avg_weight


# --------------------------------------------------------------------- #
# Gradient access

def grad_of(tensor):
    """ Get the gradient of a given tensor, make it zero if missing.
    Args:
    tensor Given instance of/deriving from Tensor
    Returns:
    Gradient for the given tensor
    """
    # Get the current gradient
    grad = tensor.grad
    if grad is not None:
        return grad
    # Make and set a zero-gradient
    grad = torch.zeros_like(tensor)
    tensor.grad = grad
    return grad


def grads_of(tensors):
    """ Iterate of the gradients of the given tensors, make zero gradients if missing.
    Args:
    tensors Generator of/iterable on instances of/deriving from Tensor
    Returns:
    Generator of the gradients of the given tensors, in emitted order
    """
    return (grad_of(tensor) for tensor in tensors)

# ---------------------------------------------------------------------------- #
# "Flatten" and "relink" operations

def relink(tensors, common):
    """ "Relink" the tensors of class (deriving from) Tensor by making them point to another contiguous segment of memory.
    Args:
    tensors Generator of/iterable on instances of/deriving from Tensor, all with the same dtype
    common  Flat tensor of sufficient size to use as underlying storage, with the same dtype as the given tensors
    Returns:
    Given common tensor
    """
    # Convert to tuple if generator
    if isinstance(tensors, types.GeneratorType):
        tensors = tuple(tensors)
    # Relink each given tensor to its segment on the common one
    pos = 0
    for tensor in tensors:
        npos = pos + tensor.numel()
        tensor.data = common[pos:npos].view(*tensor.shape)
        pos = npos
    # Finalize and return
    common.linked_tensors = tensors
    return common


def flatten(tensors):
    # Convert to tuple if generator
    if isinstance(tensors, types.GeneratorType):
        tensors = tuple(tensors)
    # Common tensor instantiation and reuse
    common = torch.cat(tuple(tensor.view(-1) for tensor in tensors))
    # Return common tensor
    return relink(tensors, common)

# ---------------------------------------------------------------------------- #

def get_gradient(model):
    gradient = flatten(grads_of(model.parameters()))
    return gradient

def set_gradient(model, gradient):
    grad_old = get_gradient(model)
    grad_old.copy_(gradient)

def get_gradient_values(model):
    gradient = torch.cat([torch.reshape(param.grad, (-1,)) for param in model.parameters()]).clone().detach()
    return gradient

def set_gradient_values(model, gradient):
    cur_pos = 0
    for param in model.parameters():
        param.grad = torch.reshape(torch.narrow(gradient, 0, cur_pos, param.nelement()), param.size()).clone().detach()
        cur_pos = cur_pos + param.nelement()

def get_parameter_values(model):
    parameter = torch.cat([torch.reshape(param.data, (-1,)) for param in model.parameters()]).clone().detach()
    return parameter

def set_parameter_values(model, parameter):
    cur_pos = 0
    for param in model.parameters():
        param.data = torch.reshape(torch.narrow(parameter, 0, cur_pos, param.nelement()), param.size()).clone().detach()
        cur_pos = cur_pos + param.nelement()
# ---------------------------------------------------------------------------- #


