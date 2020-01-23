import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from utils.utils import create_missing_folders

def plot_performance(loss_total, accuracy, labels, results_path, filename="NoName", verbose=0, std_loss=None, std_accuracy=None):
    """

    :param loss_total:
    :param loss_labelled:
    :param loss_unlabelled:
    :param accuracy:
    :param labels:
    :param results_path:
    :param filename:
    :param verbose:
    :return:
    """
    fig2, ax21 = plt.subplots()
    n = list(range(len(accuracy["train"])))
    try:
        ax21.plot(loss_total["train"], 'b-', label='Train total loss:' + str(len(labels["train"])))  # plotting t, a separately
        ax21.plot(loss_total["valid"], 'g-', label='Valid total loss:' + str(len(labels["valid"])))  # plotting t, a separately
        #ax21.plot(values["valid"], 'r-', label='Test:' + str(len(labels["valid"])))  # plotting t, a separately
    except:
        ax21.plot(loss_total["train"], 'b-', label='Train total loss:')  # plotting t, a separately
        ax21.plot(loss_total["valid"], 'g-', label='Valid total loss:')  # plotting t, a separately
    if std_accuracy is not None:
        ax21.errorbar(x=n, y=loss_total["train"], yerr=[np.array(std_loss["train"]), np.array(std_loss["train"])],
                      c="b", label='Train')  # plotting t, a separately
    if std_accuracy is not None:
        ax21.errorbar(x=n, y=loss_total["valid"], yerr=[np.array(std_loss["valid"]), np.array(std_loss["valid"])],
                      c="g", label='Valid')  # plotting t, a separately

    ax21.set_xlabel('epochs')
    ax21.set_ylabel('Loss')
    handles, labels = ax21.get_legend_handles_labels()
    ax21.legend(handles, labels)
    ax22 = ax21.twinx()

    #colors = ["b", "g", "r", "c", "m", "y", "k"]
    # if n_list is not None:
    #    for i, n in enumerate(n_list):
    #        ax22.plot(n_list[i], '--', label="Hidden Layer " + str(i))  # plotting t, a separately
    ax22.set_ylabel('Accuracy')
    ax22.plot(accuracy["train"], 'c--', label='Train')  # plotting t, a separately
    ax22.plot(accuracy["valid"], 'k--', label='Valid')  # plotting t, a separately
    if std_accuracy is not None:
        ax22.errorbar(x=n, y=accuracy["train"], yerr=[np.array(std_accuracy["train"]), np.array(std_accuracy["train"])],
                      c="c", label='Train')  # plotting t, a separately
    if std_accuracy is not None:
        ax22.errorbar(x=n, y=accuracy["valid"], yerr=[np.array(std_accuracy["valid"]), np.array(std_accuracy["valid"])],
                      c="k", label='Valid')  # plotting t, a separately

    handles, labels = ax22.get_legend_handles_labels()
    ax22.legend(handles, labels)

    fig2.tight_layout()
    # pylab.show()
    if verbose > 0:
        print("Performance at ", results_path)
    create_missing_folders(results_path + "/plots/")
    pylab.savefig(results_path + "/plots/" + filename)
    plt.show()
    plt.close()

