
def __main__():
    from models.discriminative.artificial_neural_networks.MultiLayerPerceptron import MLP
    home_path = "/home/simon/PycharmProjects/anndementia"
    destination_folder = "hebbian_learning_ann"
    data_folder = "data"
    results_folder = "results"
    meta_destination_folder = "pandas_meta_df"
    plots_folder_path = "/".join([home_path, destination_folder, results_folder, "plots/"])

    dataset_name = "mnist_dropout"
    activation = "relu"
    early_stopping = 200
    n_epochs = 1000
    gt_input = 0
    use_conv = False  # Not applicable if not sequence (images, videos, sentences, DNA...)

    lr = 1e-3
    l1 = 0.
    l2 = 0.
    dropout = 0.0
    batch_size = 64
    is_pruning = True
    # mc = 1
    # iw = 1

    # Neurons layers
    h_dims = [16, 16]

    mlp = MLP(input_size=784, input_shape=(1, 28, 28), indices_names=list(range(784)),
              num_classes=10, h_dims=h_dims, extra_class=True, l1=l1, l2=l2,
              is_hebb_layers=True, gt_input=gt_input, is_pruning=is_pruning, dropout=dropout)

    mlp.set_configs(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                    destination_folder=destination_folder, dataset_name=dataset_name, lr=lr,
                    meta_destination_folder="meta_pandas_dataframes", csv_filename="csv_loggers")

    mlp.load_example_dataset(dataset="mnist", batch_size=batch_size,
                             extra_class=True, unlabelled_train_ds=None, normalize=True, mu=0.1307, var=0.3081,
                             labels_per_class=-1)

    mlp.set_data(labels_per_class=-1, is_example=True, extra_class=True, ignore_training_inputs=3)

    mlp.cuda()
    # dgm.vae.generate_random(False, batch_size, z1_size, [1, 28, 28])
    mlp.run(n_epochs, start_pruning=3, verbose=2, show_progress=1, hist_epoch=2, is_balanced_relu=False, all0=False)

if __name__ == "__main__":
    __main__()
