import torch


def hebb_values_transform(tensor, on_zero):
    # create a copy of the original tensor,
    # because of the way we are replacing them.
    res = tensor.clone().data
    res[res == 0] = on_zero
    return res

def hebb_array_transform(tensor, on_zero):
    # create a copy of the original tensor,
    # because of the way we are replacing them.
    res = tensor.clone().data
    for i in range(len(on_zero)):
        res[i][res[i] == 0] = on_zero[i]
    return res

def balance_relu(x, overall_mean=False):
    if overall_mean:
        mu_x = torch.mean(x)
        x = [x_i if x_i > 0. else -mu_x for x_i in x]
    else:
        mu_x = torch.mean(x, 1)
        for i, x_i in enumerate(x):
            x_i = x[i]
            x_i[x_i == 0.] = -mu_x[i]
            x[i] = x_i

    return x


def hebb_values_transform_input(tensor, on_zero):
    # create a copy of the original tensor,
    # because of the way we are replacing them.
    res = tensor.clone().data
    res[res == 0] = on_zero
    return res

def hebb_values_transform_input2(tensor, on_zero):
    # create a copy of the original tensor,
    # because of the way we are replacing them.
    res = tensor.clone().data
    res[res == 0] = on_zero
    return res


def hebb_values_transform_conv(tensor, on_zero):
    # create a copy of the original tensor,
    # because of the way we are replacing them.
    res = tensor.clone().data
    res[res == 0] = on_zero

    res = torch.sum(res, dim=2)
    res = torch.sum(res, dim=2)
    return res


def hebb_values_negative_transform(tensor, multiplier=0.1):
    # create a copy of the original tensor,
    # because of the way they are replaced.
    res = tensor.clone().data
    res[res < 0] = multiplier * res[res < 0]
    return res


def indices_h(array):
    indices1 = []
    for ind, x in enumerate(array):
        if int(x) == 1:
            indices1.append(ind)
    return indices1

def indices_matrix(matrix):
    matrix1 = []
    for a, array in enumerate(matrix):
        indices1 = []
        for ind, x in enumerate(array):
            if int(x) == 1:
                indices1.append(ind)
        matrix1.append(indices1)
    return indices1

def indices_h_conv(array):
    indices1 = []
    for ind, x in enumerate(array):
        if x == 1:
            indices1.append(ind)
    return indices1

