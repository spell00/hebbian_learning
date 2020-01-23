import torch
from torch import nn
import torch.nn.functional as F
from models.semi_supervised.utils.utils import log_sum_exp, enumerate_discrete
from utils.distributions import log_standard_categorical
import numpy as np

class ImportanceWeightedSampler(object):
    """
    Importance weighted sampler [Burda 2015] to
    be used in conjunction with SVI.
    """
    def __init__(self, mc=1, iw=1):
        """
        Initialise a new sampler.
        :param mc: number of Monte Carlo samples
        :param iw: number of Importance Weighted samples
        """
        self.mc = mc
        self.iw = iw

    def resample(self, x):
        return x.repeat(self.mc * self.iw, 1)

    def __call__(self, elbo):
        elbo = elbo.view(self.mc, self.iw, -1)
        elbo = torch.mean(log_sum_exp(elbo, dim=1, sum_op=torch.mean), dim=0)
        return elbo.view(-1)


class DeterministicWarmup(object):
    """
    Linear deterministic warm-up as described in
    [Sønderby 2016].
    """
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1/n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t


class SVI(nn.Module):
    """
    Stochastic variational inference (SVI).
    """
    base_sampler = ImportanceWeightedSampler(mc=1, iw=1)
    def __init__(self, model, labels_set, images_path, dataset_name, batch_size, auxiliary, beta=1.,
                 likelihood=F.binary_cross_entropy, sampler=base_sampler, ladder=False):
        """
        Initialises a new SVI optimizer for semi-
        supervised learning.
        :param model: semi-supervised model to evaluate
        :param likelihood: p(x|y,z) for example BCE or MSE
        :param sampler: sampler for x and y, e.g. for Monte Carlo
        :param beta: warm-up/scaling of KL-term
        """
        super(SVI, self).__init__()
        self.model = model
        self.auxiliary = auxiliary
        self.likelihood = likelihood
        self.sampler = sampler
        self.ladder = ladder
        self.batch_size = batch_size

        self.images_path = images_path
        self.dataset_name = dataset_name
        self.epoch = 0
        self.labels_set = labels_set
        self.beta = beta
        self.use_conv = False
        self.last_beta = 0

    def forward(self, x, y, valid_bool, valid=False):
        self.epoch += 1
        if y is None:
            is_labelled = False
        else:
            is_labelled = True
        # Prepare for sampling
        if not self.use_conv:
            x = x.view(-1, np.prod(x.shape[1:]))
        xs, ys = (x, y)

        # Deliberate choice of tanh... # TODO show why?
        xs = torch.tanh(xs)
        # Enumerate choices of label
        if not is_labelled:
            ys = enumerate_discrete(xs, self.model.num_classes)
            xs = xs.repeat(self.model.num_classes, 1)
        # Increase sampling dimension
        if self.sampler.mc * self.sampler.iw > 1:
            xs = self.sampler.resample(xs)
            ys = self.sampler.resample(ys)
        if not self.model.sylvester_flow:
            reconstruction = self.model(xs, y=ys)
        else:
            if not self.auxiliary:
                reconstruction = self.model.run_sylvester(xs, k=0, y=ys, auxiliary=False)
            else:
                reconstruction = self.model.run_sylvester_dgm(xs, k=0, y=ys, auxiliary=True)
        try:
            likelihood = -torch.sum(self.likelihood(reconstruction, xs, reduce=False), dim=-1)
        except:
            try:
                likelihood = -torch.sum(self.likelihood(reconstruction[0], xs, reduce=False), dim=-1)
            except:
                likelihood = -torch.sum(self.likelihood(reconstruction[0].view(-1), xs, reduce=False), dim=-1)

        # p(y)
        prior = -log_standard_categorical(ys)

        # Equivalent to -L(x, y)
        if not valid:
            try:
                beta = next(self.beta)
                self.last_beta = beta
                elbo = likelihood + prior - beta * self.model.kl_divergence
            except:
                elbo = likelihood + prior - self.model.kl_divergence
        else:
            elbo = likelihood + prior - self.last_beta * self.model.kl_divergence

        L = self.sampler(elbo)

        if is_labelled:
            del elbo, xs, ys, prior, likelihood
            return -torch.mean(L), reconstruction

        logits = self.model.classify(x, valid_bool=valid_bool)
        L = L.view_as(logits.t()).t()

        # Calculate entropy H(q(y|x)) and sum over all labels
        H = -torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
        L = torch.sum(torch.mul(logits, L), dim=-1)

        # Equivalent to -U(x)
        U = L + H
        del L, H, logits, reconstruction, elbo
        return -torch.mean(U), None