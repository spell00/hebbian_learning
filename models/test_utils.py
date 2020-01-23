
from geoParser import geoParser
def get_example_merged_dataset(
    geo_ids = ["GSE33000"],
    nvidia = "yes",
    home_path = "/Users/simonpelletier/",
    results_folder = "results",
    data_folder = "data",
    destination_folder = "hebbian_learning_ann",
    initial_lr = 1e-4,
    init = "he_uniform",
    n_epochs = 10,
    batch_size = 8,
    hidden_size = 128,
    nrep = 10,
    activation = "relu",
    n_hidden = 128,
    l1 = 1e-4,
    l2 = 1e-4,
    silent = 0,
    is_load_from_disk = True):

    data_destination_folder = home_path + "/" + data_folder + "/" + destination_folder
    results_destination_folder = home_path + "/" + results_folder
    print('Data destination:', data_destination_folder)
    print('Results destination:', results_destination_folder)
    g = geoParser(destination_folder=data_destination_folder, initial_lr=initial_lr, init=init, n_epochs=n_epochs,
                  batch_size=batch_size, hidden_size=hidden_size, silent=silent)
    g.getGEO(geo_ids, is_load_from_disk=is_load_from_disk)
    # g.translate(geo_ids[0])
    g.merge_datasets()
    return g

