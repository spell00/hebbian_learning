import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import sin, cos, random


class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """
    def reparametrize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul(std, epsilon)

        return z

class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return self.reparametrize(mu, log_var), mu, log_var


#from https://raw.githubusercontent.com/musyoku/adversarial-autoencoder/master/aae/sampler.py
def onehot_categorical(batchsize, num_labels):
    y = np.zeros((batchsize, num_labels), dtype=np.float32)
    indices = np.random.randint(0, num_labels, batchsize)
    for b in range(batchsize):
        y[b, indices[b]] = 1
    return y


def uniform(batchsize, ndim, minv=-1, maxv=1):
    return np.random.uniform(minv, maxv, (batchsize, ndim)).astype(np.float32)


def gaussian(batchsize, ndim, mean=0, var=1):
    return np.random.normal(mean, var, (batchsize, ndim)).astype(np.float32)


def gaussian_mixture(batchsize, ndim, num_labels):
    if ndim % 2 != 0:
        raise Exception("ndim must be a multiple of 2.")

    def sampling(x, y, label, num_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(num_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, ndim // 2))
    y = np.random.normal(0, y_var, (batchsize, ndim // 2))
    z = np.empty((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sampling(x[batch, zi], y[batch, zi], random.randint(0, num_labels - 1), num_labels)
    return z


def supervised_gaussian_mixture(batchsize, ndim, label_indices, num_labels):
    if ndim % 2 != 0:
        raise Exception("ndim must be a multiple of 2.")

    def sampling(x, y, label, num_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(num_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, ndim // 2))
    y = np.random.normal(0, y_var, (batchsize, ndim // 2))
    z = np.empty((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sampling(x[batch, zi], y[batch, zi], label_indices[batch], num_labels)
    return z


def swiss_roll(batchsize, ndim, num_labels):

    def sampling(label, num_labels):
        uni = np.random.uniform(0.0, 1.0) / float(num_labels) + float(label) / float(num_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sampling(random.randint(0, num_labels - 1), num_labels)
    return z


def supervised_swiss_roll(batchsize, ndim, label_indices, num_labels):
    def sampling(label, num_labels):
        uni = np.random.uniform(0.0, 1.0) / float(num_labels) + float(label) / float(num_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sampling(label_indices[batch], num_labels)
    return z

class GaussianMerge(GaussianSample):
    """
    Precision weighted merging of two Gaussian
    distributions.
    Merges information from z into the given
    mean and log variance and produces
    a sample from this new distribution.
    """
    def __init__(self, in_features, out_features):
        super(GaussianMerge, self).__init__(in_features, out_features)

    def forward(self, z, mu1, log_var1):
        # Calculate precision of each distribution
        # (inverse variance)
        mu2 = self.mu(z)
        log_var2 = F.softplus(self.log_var(z))
        precision1, precision2 = (1/torch.exp(log_var1), 1/torch.exp(log_var2))

        # Merge distributions into a single new
        # distribution
        mu = ((mu1 * precision1) + (mu2 * precision2)) / (precision1 + precision2)

        var = 1 / (precision1 + precision2)
        log_var = torch.log(var + 1e-8)

        return self.reparametrize(mu, log_var), mu, log_var


class GumbelSoftmax(Stochastic):
    """
    Layer that represents a sample from a categorical
    distribution. Enables sampling and stochastic
    backpropagation using the Gumbel-Softmax trick.
    """
    def __init__(self, in_features, out_features, n_distributions):
        super(GumbelSoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_distributions = n_distributions

        self.logits = nn.Linear(in_features, n_distributions*out_features)

    def forward(self, x, tau=1.0):
        logits = self.logits(x).view(-1, self.n_distributions)

        # variational distribution over categories
        softmax = F.softmax(logits, dim=-1) #q_y
        sample = self.reparametrize(logits, tau).view(-1, self.n_distributions, self.out_features)
        sample = torch.mean(sample, dim=1)

        return sample, softmax

    def reparametrize(self, logits, tau=1.0):
        epsilon = Variable(torch.rand(logits.size()), requires_grad=False)

        if logits.is_cuda:
            epsilon = epsilon.cuda()

        # Gumbel distributed noise
        gumbel = -torch.log(-torch.log(epsilon+1e-8)+1e-8)
        # Softmax as a continuous approximation of argmax
        y = F.softmax((logits + gumbel)/tau, dim=1)
        return y
