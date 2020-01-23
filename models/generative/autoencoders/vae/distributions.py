# Adaptation of https://github.com/casperkaae/parmesan/blob/master/parmesan/distributions.py
# for pytorch

import math
import torch.tensor as T
from utils.utils import remove_nans_Variable
c = - 0.5 * math.log(2 * math.pi)

def log_normal(x, mean, std, eps=0.0):
    """
    Compute log pdf of a Gaussian distribution with diagonal covariance, at values x.
    Variance is parameterized as standard deviation.
        .. math:: \log p(x) = \log \mathcal{N}(x; \mu, \sigma^2I)

    Parameters
    ----------
    x : torch tensor
        Values at which to evaluate pdf.
    mean : torch tensor
        Mean of the Gaussian distribution.
    std : torch tensor
        Standard deviation of the diagonal covariance Gaussian.
    eps : float
        Small number added to standard deviation to avoid NaNs.
    Returns
    -------
    torch tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    See also
    --------
    log_normal1 : using variance parameterization
    log_normal2 : using log variance parameterization
    """
    std += eps
    return c - T.log(T.abs_(std)) - (x - mean) ** 2 / (2 * std ** 2)


def log_normal1(x, mean, var, eps=0.0):
    """
    Compute log pdf of a Gaussian distribution with diagonal covariance, at values x.
    Variance is parameterized as variance rather than standard deviation.
        .. math:: \log p(x) = \log \mathcal{N}(x; \mu, \sigma^2I)

    Parameters
    ----------
    x : torch tensor
        Values at which to evaluate pdf.
    mean : torch tensor
        Mean of the Gaussian distribution.
    var : torch tensor
        Variance of the diagonal covariance Gaussian.
    eps : float
        Small number added to variance to avoid NaNs.
    Returns
    -------
    torch tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    See also
    --------
    log_normal : using standard deviation parameterization
    log_normal2 : using log variance parameterization
    """
    var += eps
    return c - T.log(var) / 2 - (x - mean) ** 2 / (2 * var)


def log_normal2(x, mean, log_var, eps=0.0):
    """
    Compute log pdf of a Gaussian distribution with diagonal covariance, at values x.
    Variance is parameterized as log variance rather than standard deviation, which ensures :math:`\sigma > 0`.
        .. math:: \log p(x) = \log \mathcal{N}(x; \mu, \sigma^2I)

    Parameters
    ----------
    x : torch tensor
        Values at which to evaluate pdf.
    mean : torch tensor
        Mean of the Gaussian distribution.
    log_var : torch tensor
        Log variance of the diagonal covariance Gaussian.
    eps : float
        Small number added to denominator to avoid NaNs.
    Returns
    -------
    torch tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    See also
    --------
    log_normal : using standard deviation parameterization
    log_normal1 : using variance parameterization
    """
    return c - log_var / 2 - (x - mean) ** 2 / (2 * T.exp(log_var) + eps)


def log_stdnormal(x):
    """
    Compute log pdf of a standard Gaussian distribution with zero mean and unit variance, at values x.
        .. math:: \log p(x) = \log \mathcal{N}(x; 0, I)

    Parameters
    ----------
    x : torch tensor
        Values at which to evaluate pdf.
    Returns
    -------
    torch tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    """
    return c - x ** 2 / 2


def log_bernoulli(x, p, eps=0.0):
    """
    Compute log pdf of a Bernoulli distribution with success probability p, at values x.
        .. math:: \log p(x; p) = \log \mathcal{B}(x; p)
    Parameters
    ----------
    x : torch tensor
        Values at which to evaluate pdf.
    p : torch tensor
        Success probability :math:`p(x=1)`, which is also the mean of the Bernoulli distribution.
    eps : float
        Small number used to avoid NaNs by clipping p in range [eps;1-eps].
    Returns
    -------
    torch tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    """
    p = T.clip(p, eps, 1.0 - eps)
    return -T.nnet.binary_crossentropy(p, x)


def log_multinomial(x, p, eps=0.0):
    """
    Compute log pdf of multinomial distribution
        .. math:: \log p(x; p) = \sum_x p(x) \log q(x)
    where p is the true class probability and q is the predicted class
    probability.
    Parameters
    ----------
    x : torch tensor
        Values at which to evaluate pdf. Either an integer vector or a
        samples by class matrix with class probabilities.
    p : torch tensor
        Samples by class matrix with predicted class probabilities.
    eps : float
        Small number used to avoid NaNs by offsetting p.
    Returns
    -------
    torch tensor
        Element-wise log probability.
    """
    p += eps
    return -T.nnet.categorical_crossentropy(p, x)


def kl_normal1_stdnormal(mean, var, eps=0.0):
    """
    Closed-form solution of the KL-divergence between a Gaussian parameterized
    with diagonal variance and a standard Gaussian.
    .. math::
        D_{KL}[\mathcal{N}(\mu, \sigma^2 I) || \mathcal{N}(0, I)]
    Parameters
    ----------
    mean : torch tensor
        Mean of the diagonal covariance Gaussian.
    var : torch tensor
        Variance of the diagonal covariance Gaussian.
    eps : float
        Small number added to variance to avoid NaNs.
    Returns
    -------
    torch tensor
        Element-wise KL-divergence, this has to be summed when the Gaussian distributions are multi-variate.

    See also
    --------
    kl_normal2_stdnormal : using log variance parameterization
    """
    var += eps
    return -0.5 * (1 + T.log(var) - mean ** 2 - var)


def kl_normal2_stdnormal(mean, log_var):
    """
    Compute closed-form solution to the KL-divergence between a Gaussian parameterized
    with diagonal log variance and a standard Gaussian.
    In the setting of the variational autoencoder, when a Gaussian prior and diagonal Gaussian
    approximate posterior is used, this analytically integrated KL-divergence term yields a lower variance
    estimate of the likelihood lower bound compared to computing the term by Monte Carlo approximation.
        .. math:: D_{KL}[q_{\phi}(z|x) || p_{\theta}(z)]
    See appendix B of [KINGMA]_ for details.
    Parameters
    ----------
    mean : torch tensor
        Mean of the diagonal covariance Gaussian.
    log_var : torch tensor
        Log variance of the diagonal covariance Gaussian.
    Returns
    -------
    torch tensor
        Element-wise KL-divergence, this has to be summed when the Gaussian distributions are multi-variate.
    See also
    --------
    kl_normal1_stdnormal : using variance parameterization
    References
    ----------
        ..  [KINGMA] Kingma, Diederik P., and Max Welling.
            "Auto-Encoding Variational Bayes."
            arXiv preprint arXiv:1312.6114 (2013).
    """
    return -0.5 * (1 + log_var - mean ** 2 - T.exp(log_var))


def kl_normal1_normal1(mean1, var1, mean2, var2, eps=0.0):
    """
    Compute closed-form solution to the KL-divergence between two Gaussians parameterized
    with diagonal variance.
    Parameters
    ----------
    mean1 : torch tensor
        Mean of the q Gaussian.
    var1 : torch tensor
        Variance of the q Gaussian.
    mean2 : torch tensor
        Mean of the p Gaussian.
    var2 : torch tensor
        Variance of the p Gaussian.
    eps : float
        Small number added to variances to avoid NaNs.
    Returns
    -------
    torch tensor
        Element-wise KL-divergence, this has to be summed when the Gaussian distributions are multi-variate.
    See also
    --------
    kl_normal2_normal2 : using log variance parameterization
    """
    var1 += eps
    var2 += eps
    return 0.5 * T.log(var2 / var1) + (var1 + (mean1 - mean2) ** 2) / (2 * var2) - 0.5


def kl_normal2_normal2(mean1, log_var1, mean2, log_var2, eps=0.0):
    """
    Compute closed-form solution to the KL-divergence between two Gaussians parameterized
    with diagonal log variance.
    .. math::
       D_{KL}[q||p] &= -\int p(x) \log q(x) dx + \int p(x) \log p(x) dx     \\
                    &= -\int \mathcal{N}(x; \mu_2, \sigma^2_2) \log \mathcal{N}(x; \mu_1, \sigma^2_1) dx
                        + \int \mathcal{N}(x; \mu_2, \sigma^2_2) \log \mathcal{N}(x; \mu_2, \sigma^2_2) dx     \\
                    &= \frac{1}{2} \log(2\pi\sigma^2_2) + \frac{\sigma^2_1 + (\mu_1 - \mu_2)^2}{2\sigma^2_2}
                        - \frac{1}{2}( 1 + \log(2\pi\sigma^2_1) )      \\
                    &= \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma^2_1 + (\mu_1 - \mu_2)^2}{2\sigma^2_2} - \frac{1}{2}
    Parameters
    ----------
    mean1 : torch tensor
        Mean of the q Gaussian.
    log_var1 : torch tensor
        Log variance of the q Gaussian.
    mean2 : torch tensor
        Mean of the p Gaussian.
    log_var2 : torch tensor
        Log variance of the p Gaussian.
    eps : float
        Small number added to denominator to avoid NaNs.
    Returns
    -------
    torch tensor
        Element-wise KL-divergence, this has to be summed when the Gaussian distributions are multi-variate.
    See also
    --------
    kl_normal1_normal1 : using variance parameterization
    """
    return 0.5 * log_var2 - 0.5 * log_var1 + (T.exp(log_var1) + (mean1 - mean2) ** 2) / (
                2 * T.exp(log_var2) + eps) - 0.5


# from https://github.com/jmtomczak/vae_vpflows/blob/master/utils/distributions.py

import torch

#=======================================================================================================================
def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    probs = torch.clamp( mean, min=1e-7, max=1.-1e-7 )
    log_normal = -0.5 * ( log_var + torch.pow( x - probs, 2 ) * torch.pow( torch.exp( log_var ), -1) )
    log_normal = remove_nans_Variable(log_normal)
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

def log_Normal_standard(x, average=False, dim=1):
    log_normal = -0.5 * torch.pow( x , 2 )
    log_normal = remove_nans_Variable(log_normal)
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

def log_Bernoulli(mean, x, average=False, dim=1):
    probs = torch.clamp(mean, min=1e-7, max=1.-1e-7)
    try:
        log_bernoulli = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)
    except:
        if torch.cuda.is_available():
            x, probs = x.double(), probs.double()
        else:
            x, probs = x.cpu().double().cuda(), probs.cpu().double().cuda()
        log_bernoulli = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)

    log_bernoulli = remove_nans_Variable(log_bernoulli)
    if average:
        return torch.mean(
            torch.mean( log_bernoulli, dim )
        )
    else:
        return torch.mean(
            torch.sum( log_bernoulli, dim )
        )


import math
import torch
import torch.nn.functional as F


def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x.

    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)


def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x.

    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=-1)


def log_standard_categorical(p):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.

    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)

    return cross_entropy