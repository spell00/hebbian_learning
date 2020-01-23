import numpy as np
import matplotlib.pyplot as plt
import pylab
from utils.utils import create_missing_folders

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_performance(values, labels, n_list=None, results_path="~", filename="NoName"):
    fig2, ax21 = plt.subplots()
    plt.ylim([0., 1000.])
    ax21.plot(values["train"], 'b-', label='Train:' + str(len(labels["train"])))  # plotting t, a separately
    ax21.plot(values["valid"], 'g-', label='Valid:' + str(len(labels["valid"])))  # plotting t, a separately
    ax21.plot(values["valid"], 'r-', label='Test:' + str(len(labels["valid"])))  # plotting t, a separately
    ax21.set_xlabel('epochs')
    ax21.set_ylabel('Accuracy')
    handles, labels = ax21.get_legend_handles_labels()
    ax21.legend(handles, labels)
    ax22 = ax21.twinx()
    #colors = ["b", "g", "r", "c", "m", "y", "k"]
    if n_list is not None:
        for i, n in enumerate(n_list):
            ax22.plot(n_list[i], '--', label="Hidden Layer " + str(i))  # plotting t, a separately
    ax22.set_ylabel('#Neurons')
    handles, labels = ax22.get_legend_handles_labels()
    ax22.legend(handles, labels)

    fig2.tight_layout()
    # pylab.show()
    pylab.savefig(results_path + "/plots/hnet/" + filename)
    create_missing_folders(results_path + "/plots/hnet/")
    plt.close()


def plot_losses(losses, labels, n_neurons=None, results_path="~",filename="NoName"):
    filename = "_".join([filename, "loss.png"])
    create_missing_folders(results_path + "/plots/hnet/")
    fig, ax1 = plt.subplots()
    plt.ylim([0., 1000.])

    ax1.plot(losses, 'g-.', label='train')  # plotting t, a separately

    ax1.set_xlabel('epochs')
    ax1.set_ylabel('Loss')

    # ax1.tick_params('y')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    if n_neurons is not None:
        ax22 = ax1.twinx()
        for i, n in enumerate(n_neurons):
            ax22.plot(n_neurons[i], '--', label="Hidden Layer " + str(i))  # plotting t, a separately
        ax22.set_ylabel('#Neurons')
        handles, labels = ax22.get_legend_handles_labels()
        ax22.legend(handles, labels)

    fig.tight_layout()
    # pylab.show()
    pylab.savefig(results_path + "/plots/hnet/" + filename)
    plt.close()

