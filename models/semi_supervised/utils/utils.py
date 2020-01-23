import torch
from torch.autograd import Variable
import numpy as np

def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """
    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return Variable(generated.float())

def one_hot(k):
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """
    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y
    return encode

def onehot(n, pos):
    return np.array([0 if i != pos else 1 for i in range(n)])


def onehot_array(pos_array, n):
    return np.array([onehot(n, pos) for pos in pos_array])


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max

def rename_model(model_name, z2_size, warmup, z1_size):
    model_name = model_name
    if model_name == 'vae_HF':
        number_combination = 0
    elif model_name == 'vae_ccLinIAF':
        number_of_flows = 1

    if model_name == 'vae_HF':
        model_name = model_name + '(T_' + str(number_of_flows) + ')'
    elif model_name == 'vae_ccLinIAF':
        model_name = model_name + '(K_' + str(number_combination) + ')'

    model_name = model_name + '_wu(' + str(warmup) + ')' + '_z1_' + str(z1_size)

    if z2_size > 0:
        model_name = model_name + '_z2_' + str(z2_size)
    return model_name

