from __future__ import print_function

import torch
import torch.utils.data as data_utils

import numpy as np

from scipy.io import loadmat
import os

import pickle
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def load_static_mnist(batch_size,test_batch_size):
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    x_train = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    x_val = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    x_test = lines_to_np_array(lines).astype('float32')

    # shuffle train data
    np.random.shuffle(x_train)

    # idle y's
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )
    y_test = np.zeros( (x_test.shape[0], 1) )

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=test_batch_size, shuffle=False)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=test_batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


# ======================================================================================================================
def load_dynamic_mnist(batch_size, test_batch_size):
    from keras.datasets import mnist
    # loading data from Keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # preparing data
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    y_test = np.array(y_test, dtype=int)

    x_val = x_train[50000:60000]
    y_val = np.array(y_train[50000:60000], dtype=int)
    x_train = x_train[0:50000]
    y_train = np.array(y_train[0:50000], dtype=int)

    # binarize
    np.random.seed(777)
    x_val = np.random.binomial(1, x_val)
    x_test = np.random.binomial(1, x_test)

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=test_batch_size, shuffle=False)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ======================================================================================================================
def load_omniglot(batch_size, test_batch_size, n_validation=1345):
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')
    omni_raw = loadmat(os.path.join('datasets', 'OMNIGLOT', 'chardata.mat'))

    # train and test data
    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))

    # shuffle train data
    np.random.shuffle(train_data)

    # set train and validation data
    x_train = train_data[:-n_validation]
    validation_data = train_data[-n_validation:]

    # fixed binarization
    np.random.seed(777)
    x_val = np.random.binomial(1, validation_data)
    x_test = np.random.binomial(1, test_data)

    # idle y's
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )
    y_test = np.zeros( (x_test.shape[0], 1) )

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=test_batch_size, shuffle=False)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=test_batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


# ======================================================================================================================
def load_caltech101silhouettes(batch_size,test_batch_size):
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')
    caltech_raw = loadmat(os.path.join('datasets', 'Caltech101Silhouettes', 'caltech101_silhouettes_28_split1.mat'))

    # train, validation and test data
    x_train = reshape_data(caltech_raw['train_data'].astype('float32'))
    np.random.shuffle(x_train)
    x_val = reshape_data(caltech_raw['val_data'].astype('float32'))
    np.random.shuffle(x_val)
    x_test = reshape_data(caltech_raw['test_data'].astype('float32'))

    y_train = caltech_raw['train_labels']
    y_val = caltech_raw['val_labels']
    y_test = caltech_raw['test_labels']

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=test_batch_size, shuffle=False)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=test_batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


# ======================================================================================================================
def load_dataset(dataset_name):
    if dataset_name == 'static_mnist':
        train_loader, val_loader, test_loader = load_static_mnist()
    elif dataset_name == 'dynamic_mnist':
        train_loader, val_loader, test_loader = load_dynamic_mnist()
    elif dataset_name == 'omniglot':
        train_loader, val_loader, test_loader = load_omniglot()
    elif dataset_name == 'caltech101silhouettes':
        train_loader, val_loader, test_loader = load_caltech101silhouettes()
    else:
        raise Exception('Wrong name of the dataset!')

    return train_loader, val_loader, test_loader
