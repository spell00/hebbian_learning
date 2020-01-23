
import glob
import numpy as np
import matplotlib.pyplot as plt
import pylab
for file in glob.glob("data/*.npy"):
    print(file)
    results = np.load(file)
    running_train_accuracies = np.ndarray.tolist(results[0])
    running_valid_accuracies = np.ndarray.tolist(results[1])
    running_test_accuracies = np.ndarray.tolist(results[2])
    running_train_loss = np.ndarray.tolist(results[3])
    running_valid_loss = np.ndarray.tolist(results[4])
    running_test_loss = np.ndarray.tolist(results[5])

    attributes = file.split("_")
    augmentation = attributes[0]
    LR = attributes[2]
    dropout=0.5
    init = "he"
    Ns = attributes[1]
    layers_planes = attributes[-1].split(".npy")[0]

    filename = str(augmentation) + "_" + str(LR) + "_" + str(dropout) + "_" + init + "_" + str(Ns) + "_" + str(layers_planes)
    fig, ax1 = plt.subplots()

    ax1.plot(running_train_loss, 'g-', label='train')  # plotting t, a separately
    ax1.plot(running_valid_loss, 'b-', label='valid')  # plotting t, a separately
    ax1.plot(running_test_loss, 'r-', label='test')  # plotting t, a separately

    ax1.set_xlabel('epochs')
    ax1.set_ylabel('Loss')
    #ax1.tick_params('y')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)

    fig.tight_layout()
    #pylab.show()
    pylab.savefig(filename + "loss.png")
    plt.close()
    fig2, ax2 = plt.subplots()
    ax2.plot(running_train_accuracies, 'g-', label='train')  # plotting t, a separately
    ax2.plot(running_valid_accuracies, 'b-', label='valid')  # plotting t, a separately
    ax2.plot(running_test_accuracies, 'r-', label='test')  # plotting t, a separately
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('Accuracy')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)

    fig2.tight_layout()
    #pylab.show()
    pylab.savefig(filename + "accuracy.png")
    plt.close()



def imshow(img):
    import matplotlib.pyplot as plt
    import numpy as np
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

