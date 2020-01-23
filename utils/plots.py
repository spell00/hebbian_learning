# normal standard curve

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from utils.utils import create_missing_folders
from scipy.stats import halfnorm
import scipy

def normal_curve(ax, mu=0., sigma=1.):
    try:
        sigma = np.sqrt(sigma)
    except:
        print("problem with sigma", sigma)
        return
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot(x, scipy.stats.norm.pdf(x, mu, sigma))
    plt.axvline(x=mu, c="r", linewidth=1)


def half_normal_curve(ax, mu=0., sigma=1., half_mu=0.):
    x = np.linspace(0, mu + 3*sigma, 100)
    # x = np.linspace(halfnorm.ppf(0.01),
    #                halfnorm.ppf(0.99), 100)
    ax.plot(x, halfnorm.pdf(x, mu, sigma))
    plt.axvline(x=mu, c="r", linewidth=1)
    plt.axvline(x=half_mu, c="g", linewidth=1)


def histograms_hidden_layers(xs, results_path, normalized, is_mean=True, epoch=0, depth=0, activated=False,
                             mu=None, var=None, axis=0, bins=50, flat=True, neuron=None):
    ax = plt.subplot(111)
    ax.set_xlabel("Hidden value")
    ax.set_ylabel("Frequency")
    plt.title("PDF of preactivation values")

    if neuron is None:
        neurons = "all"
    else:
        neurons = "single"
        xs = xs[:, neuron]

    if is_mean:
        xs = np.mean(xs, axis=axis)
    ax.hist(xs, bins=bins, alpha=0.5, density=True)

    if mu is None and var is None:
        mean_mean = float(np.mean(xs))
        mean_var = float(np.var(xs))
    elif mu is not None and var is not None:
        mean_mean = float(mu)
        mean_var = float(var)
    else:
        print("No images saved. Both mu and var must be either None or both numpy")
        return
    normal_curve(ax, mean_mean, mean_var)
    if activated:
        plt.axvline(x=float(np.mean(xs)), c="g", linewidth=1)

    #    half_normal_curve(ax, mu, var, float(np.mean(xs)))
    destination_folder_path = "/".join((results_path, "layers_histograms", "depth_"+str(depth),
                                        "activated_"+ str(activated), "normalized_"+str(normalized))) + "/"
    create_missing_folders(destination_folder_path)
    destination_file_path = destination_folder_path + "Hidden_values_hist_" + str(epoch) + "_activated"+ \
                            str(activated) + "_normalized" + str(normalized) + "_mean" + str(is_mean) + "_flat"\
                            + str(flat) + "_" + neurons + "neurons.png"
    plt.savefig(destination_file_path)
    plt.close()

