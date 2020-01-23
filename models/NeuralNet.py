from __future__ import print_function
import numpy as np
import pylab
import matplotlib.pyplot as plt
from utils.utils import create_missing_folders, make_classes
import torch
import torch.nn as nn
from utils.data_split import validation_split
import torchvision.datasets as dset

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def plot_performance(values, labels, n_list=None, results_path="~", filename="NoName"):
    fig2, ax21 = plt.subplots()

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


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.extra_class = None

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        if torch.cuda.is_available():
            has_cuda = True
            torch.cuda.manual_seed(1)
        else:
            has_cuda = False

        self.lr = None
        self.mom = None
        self.optimizer = None
        self.has_cuda = has_cuda
        self.input_size = {}
        self.x_train = None
        self.x_valid = None
        self.x_test = None
        self.y_test = None
        self.y_train = None
        self.y_valid = None
        self.classes_train = None
        self.classes_valid = None
        self.classes_test = None
        self.labels_train = None
        self.labels_valid = None
        self.labels_test = None
        self.labels = None
        self.labels_set = None
        self.num_classes = None
        self.train_loader_unlabelled = None

        # Empty lists
        self.accuracy_training_array = []
        self.accuracy_valid_array = []
        self.losses_training_array = []
        self.losses_valid_array = []
        self.max_valid_accuracies = []
        self.max_valid_epochs = []
        self.min_valid_loss = []
        self.min_valid_loss_epochs = []

        self.model = None
        self.meta_df = {}
        self.epoch = 0
        self.init = None
        self.batch_size = None
        self.nrep = None
        self.classes_train = None
        self.classes_test = None

        self.dataset_name = None
        self.filename = None
        self.csv_filename = None

        # Folder names
        self.results_folder = None
        self.destination_folder = None
        self.data_folder = None
        self.meta_destination_folder = None
        # Paths
        self.home_path = None
        self.results_path = None
        self.models_path = None
        self.model_history_path = None

        self.csv_logger_path = None
        self.data_folder_path = None
        self.meta_destination_path = None

        # Lists
        self.kernels_pooling = []
        self.running_losses = []
        self.running_accuracies = []

        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.test_accuracies = []
        self.invalid_list = []

        self.toKeep_indices = []

        self.classes = None
        self.random_training = None
        self.classes_not_in_training = None
        self.random_valid = None
        self.meta_data_folder_path = None
        self.ratio_training = None
        self.ratio_valid = None

    def set_configs(self, home_path, dataset_name, results_folder="results", data_folder = "data", destination_folder="hebbian_learning_ann",
                    meta_destination_folder = "meta_pandas_dataframes", csv_filename = "csv_loggers", lr=1e-3, mom=0,
                    is_unlabelled=True):

        # Hyper-parameters
        self.lr = float(lr)
        self.mom = mom
        self.is_unlabelled = is_unlabelled

        # Files names
        if type(dataset_name) == list:
            names = [name for name in dataset_name]
            dataset_name = "_".join(names)
        self.dataset_name = dataset_name
        self.filename = dataset_name + '_history'
        self.csv_filename = csv_filename

        # Folder names
        self.results_folder = results_folder
        self.destination_folder = destination_folder
        self.data_folder = data_folder
        self.meta_destination_folder = meta_destination_folder
        # Paths
        self.home_path = home_path
        self.results_path = "/".join([self.home_path,  self.destination_folder, self.results_folder])
        self.models_path = "/".join([self.results_path, "models"])
        self.model_history_path = self.models_path + "/history/"

        self.csv_logger_path = "/".join([self.results_path, csv_filename])
        self.data_folder_path = "/".join([home_path, self.destination_folder, self.data_folder])
        self.meta_data_folder_path = "/".join([self.data_folder_path, self.meta_destination_folder])
        #create_missing_folders(self.csv_logger_path)
        #create_missing_folders(self.models_path)
        #create_missing_folders(self.meta_data_folder_path)
        #create_missing_folders(self.model_history_path)

        self.filename = self.dataset_name + '_history'

    def import_dataframe(self, dataframe, batch_size, labelled, ratio_valid=0.1, ratio_test=0.1):
        self.batch_size = batch_size
        self.indices_names = list(dataframe.index)
        self.ratio_test = ratio_test
        self.ratio_valid = ratio_valid
        self.meta_df[labelled] = dataframe
        self.input_size = self.meta_df[labelled].shape[0]
        self.input_shape = [self.input_size]
        if labelled:
            self.labels = list(self.meta_df[labelled].columns)

    def load_dataframe(self, labelled):
        self.meta_df[labelled] = np.load(self.meta_data_folder_path + '/' + self.dataset_name + "_labelled" + labelled)
        self.meta_df[labelled][np.isnan(self.meta_df)] = 0
        self.input_size = self.meta_df.shape[0]

    def create_classes(self):
        import random
        self.classes = make_classes(self.labels)
        indices = list(range(len(self.classes)))
        random.shuffle(indices)
        num_training = int(len(indices) * self.ratio_training)
        num_valid = int(len(indices) * self.ratio_valid)
        self.random_training = indices[:num_training]
        self.random_valid = indices[num_training:(num_training+num_valid)]
        self.random_test = indices[(num_training+num_valid):]

        self.classes_train = self.classes[self.random_training]
        self.classes_valid = self.classes[self.random_valid]
        self.classes_test = self.classes[self.random_test]

    def split_data(self):
        x_values = np.transpose(self.meta_df[True].values)
        if self.is_unlabelled:
            self.unlabelled_x_train = np.transpose(self.meta_df[False].values)
        self.x_train = x_values[self.random_training, :]
        self.x_valid = x_values[self.random_valid, :]
        self.x_test = x_values[self.random_test, :]
        self.labels_train = np.array(self.labels)[self.random_training]
        self.labels_valid = np.array(self.labels)[self.random_valid]
        self.labels_test = np.array(self.labels)[self.random_test]

        if self.is_unlabelled:
            print(self.unlabelled_x_train.shape[0], 'unlabelled train samples')
        print(self.x_train.shape[0], 'train samples')
        print(self.x_valid.shape[0], 'valid samples')
        print(self.x_test.shape[0], 'valid samples')
        print(len(self.classes), 'total samples')

    def set_data(self, labels_per_class, ratio_training=0.8, ratio_valid=0.1, is_example=False,
                 is_split=True, extra_class=False,  has_unlabelled_samples=False, ignore_training_inputs=0,
                 add_noise_samples=False, is_custom_data=False):
        import torch.utils.data
        if self.input_size is None:
            if len(self.input_shape) == 1:
                self.input_size = int(self.input_shape[0])
            elif len(self.input_shape) == 2:
                self.input_size = int(self.input_shape[1])
            elif len(self.input_shape) == 3:
                self.input_size = int(self.input_shape[0]) * int(self.input_shape[1]) * int(self.input_shape[2])
            elif len(self.input_shape) == 4:
                self.input_size = int(self.input_shape[1]) * int(self.input_shape[2]) * int(self.input_shape[3])

        self.ratio_training = ratio_training
        self.ratio_valid = ratio_valid
        self.ratio_test = 1 - (ratio_valid + ratio_training)
        self.extra_class = extra_class
        self.labels_set = list(set(self.labels))
        if self.extra_class:
            self.labels_set = list(self.labels_set) + ["N/A"]
            na = 1
        else:
            na = 0
            self.labels_set = list(set(self.labels))
        self.num_classes = len(self.labels_set)
        self.create_classes()
        if not is_example and is_split:
            self.split_data()
        if self.labels_train is None:
            self.labels_train = self.classes_train
        if self.labels_valid is None:
            self.labels_valid = self.classes_valid
        if self.labels_test is None:
            self.labels_test = self.classes_test
        rows_to_keep = [True if label in self.labels_set[ignore_training_inputs:self.num_classes+na] else False for label in self.labels_train]

        # This is done just on training, testing has all the inputs previously ignored
        self.y_train = to_categorical(self.classes_train, self.num_classes)
        self.y_valid = to_categorical(self.classes_valid, self.num_classes)
        try:
            self.x_train = self.x_train.train_data[rows_to_keep, :]
            self.y_train = self.y_train[rows_to_keep, :]
        except:
            pass
            #self.x_train = self.x_train[rows_to_keep]

        self.y_test = to_categorical(self.classes_test, self.num_classes)
        if not is_example and not is_custom_data:
            if self.is_unlabelled:
                self.y_unlabelled = np.array([-1] * len(self.unlabelled_x_train))
                unlabelled_train = torch.utils.data.TensorDataset(torch.from_numpy(self.unlabelled_x_train),
                                                              torch.from_numpy(self.y_unlabelled))
            else:
                unlabelled_train = None
            train = torch.utils.data.TensorDataset(torch.from_numpy(self.x_train), torch.from_numpy(self.y_train))
            valid = torch.utils.data.TensorDataset(torch.from_numpy(self.x_valid), torch.from_numpy(self.y_valid))
            test = torch.utils.data.TensorDataset(torch.from_numpy(self.x_test), torch.from_numpy(self.y_test))

            self.make_loaders(train, valid, test, labels_per_class, unlabelled_train_ds=unlabelled_train,
                              unlabelled_samples=has_unlabelled_samples)


    def load_local_dataset(self, root_train, root_valid, root_test, n_classes, batch_size=128, labels_per_class=-1, extra_class=False,
                             unlabelled_train_ds=True, normalize=True, mu=0.1307, var=0.3081):
        from models.semi_supervised.utils.utils import one_hot
        import torchvision.transforms as transforms
        self.batch_size = batch_size
        self.extra_class = extra_class
        dataset = dset.ImageFolder
        self.input_shape = [1, 28, 28]
        if self.extra_class:
            n_classes += 1
        download = True  # download MNIST dataset or not

        # Normalize is for MNIST dataset
        if normalize:
            flatten = lambda x: \
                transforms.Compose([transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((mu,), (var,))])(x).view(-1)

        else:
            flatten = lambda x: \
                transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])(x).view(-1)

        train_ds = dataset(root=root_train, transform=flatten,
                           target_transform=one_hot(n_classes))
        valid_ds = dataset(root=root_valid, transform=flatten,
                           target_transform=one_hot(n_classes))
        test_ds = dataset(root=root_test, transform=flatten)
        self.labels_train = [x[1] for x in train_ds.samples]
        self.labels_valid = [x[1] for x in valid_ds.samples]
        self.labels = self.labels_train
        print("NUMBER OF LABELS", len(self.labels_train))
        self.labels_set = list(set(self.labels))
        if extra_class:
            self.labels_set += ["N/A"]
        self.num_classes = len(self.labels_set)
        self.x_train = train_ds
        self.x_valid = valid_ds
        self.x_test = test_ds

        self.make_loaders(train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, labels_per_class=labels_per_class,
                          unlabelled_train_ds=None, unlabelled_samples=unlabelled_train_ds)


    def load_example_dataset(self, dataset, valid_prop=0.5, batch_size=128, labels_per_class=-1, extra_class=False,
                             unlabelled_train_ds=None, normalize=True, mu=0.1307, var=0.3081, unlabelled_samples=False):
        from models.semi_supervised.utils.utils import one_hot
        import torchvision.transforms as transforms
        self.batch_size = batch_size
        self.extra_class = extra_class
        n_classes = 0
        if dataset == "mnist":
            dataset = dset.MNIST
            n_classes = 10
            self.input_shape = [1, 28, 28]
        elif dataset == "cifar10":
            dataset = dset.CIFAR10
            n_classes = 10
            self.input_shape = [3, 32, 32]
        elif dataset == "cifar100":
            dataset = dset.CIFAR100
            n_classes = 100
            self.input_shape = [3, 32, 32]
        else:
            print("Not a valid dataset (or not implemented)")
        if self.extra_class:
            n_classes += 1
        root = "~/data"
        download = True  # download MNIST dataset or not

        # Normalize is for MNIST dataset
        if normalize:
            flatten = lambda x: \
                transforms.Compose([transforms.ToTensor(), transforms.Normalize((mu,), (var,))])(x).view(-1)

        else:
            flatten = lambda x: \
                transforms.Compose([transforms.ToTensor()])(x).view(-1)

        train_ds = dataset(root=root, train=True, transform=flatten, download=download,
                           target_transform=one_hot(n_classes))
        test_ds = dataset(root=root, train=False, transform=flatten, download=download,
                          target_transform=one_hot(n_classes))
        try:
            self.labels_train = train_ds.train_labels.numpy().tolist()
            self.labels_valid = test_ds.test_labels.numpy().tolist()
        except:
            self.labels_train = train_ds.train_labels
            self.labels_valid = test_ds.test_labels
        self.labels = self.labels_train
        self.labels_set = list(set(self.labels))
        if extra_class:
            self.labels_set += ["N/A"]
        self.num_classes = len(self.labels_set)
        valid_prop = valid_prop
        valid_ds, test_ds = validation_split(test_ds, valid_prop)
        self.x_train = train_ds
        self.x_valid = valid_ds
        self.x_test = test_ds

        self.make_loaders(train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds, labels_per_class=labels_per_class,
                          unlabelled_train_ds=unlabelled_train_ds, unlabelled_samples=unlabelled_samples)

    def make_loaders(self, train_ds, valid_ds, test_ds, labels_per_class, unlabelled_train_ds, unlabelled_samples=False):
        from functools import reduce
        from operator import __or__
        from torch.utils.data.sampler import SubsetRandomSampler
        def get_sampler(labels, n=None):
            # Only choose digits in n_labels
            labels_set = set(labels)
            (indices,) = np.where(reduce(__or__, [labels == i for i in labels_set]))

            # Ensure uniform distribution of labels
            np.random.shuffle(indices)
            indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in labels_set])

            indices = torch.from_numpy(indices)
            sampler = SubsetRandomSampler(indices)
            return sampler
        if unlabelled_train_ds is not None:
            self.train_loader = torch.utils.data.DataLoader(
                dataset=train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )

            self.train_loader_unlabelled = torch.utils.data.DataLoader(
                dataset=unlabelled_train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )

        else:
            if labels_per_class > 0:
                print("Limited number of labels:", labels_per_class)
                self.train_loader = torch.utils.data.DataLoader(
                    dataset=train_ds,
                    batch_size=self.batch_size,
                    sampler=get_sampler(np.array(self.labels_train), labels_per_class),
                    num_workers=0,
                    drop_last=True
                )
                if unlabelled_samples:
                    self.train_loader_unlabelled = torch.utils.data.DataLoader(
                        dataset=train_ds,
                        batch_size=self.batch_size,
                        sampler=get_sampler(np.array(self.labels_train)),
                        num_workers=0,
                        drop_last=True
)
            else:
                print("No unlabelled data")
                self.train_loader = torch.utils.data.DataLoader(
                    dataset=train_ds,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0,
                    drop_last=True
                )
                if unlabelled_samples:
                    self.train_loader_unlabelled = torch.utils.data.DataLoader(
                        shuffle=True,
                        dataset=unlabelled_train_ds,
                        batch_size=self.batch_size,
                        num_workers=0,
                        drop_last=True

                    )

        print("self.train_loader", len(self.train_loader))
        print("self.train_loader", self.train_loader.batch_size)
        print("self.train_loader", self.train_loader.sampler)
        self.valid_loader = torch.utils.data.DataLoader(
            dataset=valid_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        if test_ds is not None:
            self.test_loader = torch.utils.data.DataLoader(
                dataset=test_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True
)

    def forward(self, *args):
        pass

    def set_init(self, init="glorot", ask=False):
        he_choices = ["he", "He", "Kaiming", "kaiming", "Kaiming_uniform", "kaiming_uniform", "kaiming uniform",
                      "Kaiming uniform", "Kaiming Uniform", "Kaiming_Uniform"]
        glorot_choices = ["glorot", "Glorot", "Xavier", "xavier", "xavier_uniform", "xavier uniform", "Xavier_uniform",
                          "Xavier_Uniform", "Xavier uniform", "Xavier_uniform"]
        if init in he_choices and ask:
            flag = False
            if init == "he" or init == "He" or init == "Kaiming" or init == "kaiming":
                choice = input("He (kaiming) initialization can be uniform or normal. Which one do you want? "
                               "[u/n] (default is uniform [u] ) ")
                if choice != "" and choice != "u":
                    self.init_function = torch.nn.init.kaiming_normal_
                    flag = True
            if not flag:
                self.init_function = self.init_function_
        if init in glorot_choices:
            flag = False
            if init in glorot_choices and ask:
                choice = input("Xavier (glorot) initialization can be uniform or normal. Which one do you want?"
                               " [u/n] (default is uniform [u] ) ")
                if choice != "" and choice != "u":
                    self.init_function = torch.nn.init.xavier_normal_
                    flag = True
            elif not flag:
                self.init_function = torch.nn.init.xavier_uniform_
        elif init == "uniform" or init == "Uniform":
            self.init_function = torch.nn.init.uniform_
        elif init == "constant" or init == "Constant":
            self.init_function = torch.nn.init.constant_
        elif init == "eye" or init == "Eye":
            self.init_function = torch.nn.init.eye_
        elif init == "dirac" or init == "Dirac":
            self.init_function = torch.nn.init.orthogonal_
        elif init == "orthogonal" or init == "Orthogonal":
            self.init_function = torch.nn.init.orthogonal_

    def set_optim(self, optim_type="adam"):
        if optim_type == "adam" or optim_type == "Adam":
            self.optimization = torch.optim.Adam
        elif optim_type == "SGD" or optim_type == "sgd":
            self.optimization = torch.optim.SGD
        elif optim_type == "Adagrad" or optim_type == "adagrad":
            self.optimization = torch.optim.Adagrad
        elif optim_type == "Adadelta" or optim_type == "adadelta":
            self.optimization = torch.optim.Adadelta
        elif optim_type == "SparseAdam" or optim_type == "sparseAdam":
            self.optimization = torch.optim.SparseAdam
        elif optim_type == "Adagrad" or optim_type == "adagrad":
            self.optimization = torch.optim.Adagrad
        elif optim_type == "Adamax" or optim_type == "adamax":
            self.optimization = torch.optim.Adamax
        elif optim_type == "ASGD" or optim_type == "asgd":
            self.optimization = torch.optim.SGD
        elif optim_type == "LBFGS" or optim_type == "lbfgs":
            self.optimization = torch.optim.Adagrad
        elif optim_type == "RMSprop" or optim_type == "rmsprop":
            self.optimization = torch.optim.Adagrad
        elif optim_type == "Rprop" or optim_type == "rprop":
            self.optimization = torch.optim.Adamax
        else:
            print("problem! optim_type is wrong:", optim_type)

    def save_model(self):
        # SAVING
        print("MODEL (with classifier) SAVED AT LOCATION:", self.model_history_path)
        #create_missing_folders(self.model_history_path)
        torch.save(self.state_dict(), self.model_history_path + self.model_file_name +'.state_dict')


    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def cross_validation(self, n=3):
        #TODO
        # Make the function so it would call set_data from here, then it does it n times
        return

