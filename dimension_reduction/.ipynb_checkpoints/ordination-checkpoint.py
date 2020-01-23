from utils.utils import ellipse_data, create_missing_folders
from utils.utils import ellipse_data, create_missing_folders
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

def get_colors():
    n = 6
    color = plt.cm.coolwarm(np.linspace(0.1, 0.9, n))  # This returns RGBA; convert:
    hexcolor = map(lambda rgb: '#%02x%02x%02x' % (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255),
                   tuple(color[:, 0:-1]))
    return hexcolor


def ordination2d(data_frame, ord_type, images_folder_path, dataset_name, epoch, a=0.5, verbose=0, info="none",
                 show_images=True):
    import pandas as pd
    import numpy as np
    try:
        assert type(data_frame) == pd.core.frame.DataFrame
    except:
        print("The type of the data object in pca2d has to be pandas.core.frame.DataFrame. Returning without finishing (no PCA plot was produced)")
        print(type(data_frame))
        exit()
        return
    if type(dataset_name) == list:
        names = [name for name in dataset_name]
        dataset_name = "_".join(names)

    y = np.array(data_frame.columns, dtype=str)
    classes_list = np.unique(y)
    data_frame.values[np.isnan(data_frame.values)] = 0
    ord = None
    ys = False
    if ord_type in ["pca", "PCA"]:
        ys = False
        ord = PCA(n_components=2)
    elif ord_type in ["tsne", "tSNE", "TSNE", "t-sne", "T-SNE", "t-SNE"]:
        ys = False
        ord = TSNE(n_components=2, verbose=verbose)
    elif ord_type in ["lda", "LDA", "flda", "FLDA"]:
        ys = True
        ord = LDA(n_components=2)
    elif ord_type in ["qda", "QDA"]:
        ord = QDA()
        ys = True
    else:
        exit("No ordination of that name is implemented. Exiting...")
    if ys:
        principal_components = ord.fit_transform(np.transpose(data_frame.values))
    else:
        principal_components = ord.fit_transform(np.transpose(data_frame.values), y=data_frame.columns)

    principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
    final_df = pd.concat([principal_df, pd.DataFrame(y)], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component Ordination', fontsize=20)

    colors = cm.viridis(np.linspace(0, 1, len(classes_list)))
    for t, target in enumerate(classes_list):
        indices_to_keep = final_df[0] == target
        indices_to_keep = list(indices_to_keep)
        data1 = final_df.loc[indices_to_keep, 'principal component 1']
        data2 = final_df.loc[indices_to_keep, 'principal component 2']
        try:
            assert np.sum(np.isnan(data1)) == 0 and np.sum(np.isnan(data2)) == 0
        except:
            print("Nans were detected. Please verify the DataFrame...")
            exit()
        ellipse_data(data1, data2, ax, colors[t])

        ax.scatter(data1, data2, s=20, alpha=a, linewidths=0.5, edgecolor='k', c=colors[t])
    ax.legend(classes_list)
    ax.grid()

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, classes_list)

    try:
        plt.tight_layout()
        fig.tight_layout()
    except:
        pass
    type_images_folder_path = "/".join([images_folder_path, str(ord_type), str(dataset_name)]) + "/"
    type_images_folder_path = type_images_folder_path + info + "/"

    create_missing_folders(type_images_folder_path)

    plt.savefig(type_images_folder_path + info + "_" + str(epoch) + ".png", dpi=100)
    if show_images:
        plt.show()
    plt.close(fig)


