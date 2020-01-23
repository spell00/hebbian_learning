from __future__ import print_function
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
import random
import matplotlib.pyplot as plt
import operator
from utils.utils import create_missing_folders
import torch
import torch.nn as nn

class neural_net(nn.Module):
    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            has_cuda = True
            torch.cuda.manual_seed(1)
        else:
            has_cuda = False
        self.has_cuda = has_cuda
        self.input_size = 0

    def set_configs(self, home_path, results_folder="results", data_folder = "data", destination_folder="hebbian_learning_ann",
                 dataset_name="GSE33000", meta_destination_folder = "meta_pandas_dataframes", csv_filename = "csv_loggers", lr=1e-3):

        # Hyper-parameters
        self.lr = lr


        # Files names
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
        self.meta_destination_path = "/".join([self.data_folder_path, self.meta_destination_folder])
        create_missing_folders(self.csv_logger_path)
        create_missing_folders(self.models_path)
        create_missing_folders(self.meta_destination_path)
        create_missing_folders(self.model_history_path)

        # Empty lists
        self.accuracy_training_array = []
        self.accuracy_valid_array = []
        self.losses_training_array = []
        self.losses_valid_array = []
        self.max_valid_accuracies = []
        self.max_valid_epochs = []
        self.min_valid_loss = []
        self.min_valid_loss_epochs = []
        # empty objects
        self.model = None
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.meta_df = None
        self.epoch = 0
        self.num_classes = None
        self.init = None
        self.batch_size = None
        self.nrep = None
        self.classes_train = None
        self.classes_test = None

    def import_dataframe(self, dataframe):
        self.meta_df = dataframe
        self.input_size = len(self.meta_df.index)

    def load_dataframe(self):
        self.meta_df = np.load(self.meta_data_folder_path + '/' + self.dataset_name + '_gene_expression.pickle.npy')
        self.meta_df[np.isnan(self.meta_df)] = 0
        self.input_size = len(self.meta_df.index)


    def set_data(self,ratio_training=0.9,ratio_valid=0.1):
        print('Importing data in keras model')
        labels = self.meta_df.columns
        label_set = set(list(labels))
        self.labels = list(labels)
        self.labels_set = set(list(labels))
        classes = np.copy(labels)
        self.num_classes = len(label_set)
        for index, label in enumerate(label_set):
            for lab in range(len(labels)):
                if (label == labels[lab]):
                    classes[lab] = int(index)

        random_training = np.random.choice(range(len(self.meta_df.columns)), size=(int(len(classes) * (ratio_training))), replace=False)
        classes_not_in_training = [x for x in range(len(self.meta_df.columns)) if x not in random_training]
        random_valid = np.random.choice(range(len(classes_not_in_training)), size=(int(len(classes) * (ratio_valid))), replace=False)
        classes_test = [x for x in range(len(classes)) if x not in random_training and x not in random_valid]

        try:
            assert len(classes_test) + len(random_valid) + len(random_training) == len(classes)
        except:
            print(len(classes_test),len(random_valid) , len(random_training),len(classes))
        print("shape", self.meta_df.shape)
        print("classes", classes.shape, classes[0:5])
        print("labels", labels[0:10], "LEN", len(labels))

        x_values = np.transpose(self.meta_df.values)
        self.x_train = x_values[random_training, :]
        self.x_valid = x_values[random_valid, :]
        self.x_test = x_values[classes_test, :]
        self.classes_train = classes[random_training]
        self.classes_valid = classes[random_valid]
        self.classes_test = classes[classes_test]
        self.labels_train = np.array(self.labels)[random_training]
        self.labels_valid = np.array(self.labels)[random_valid]
        self.labels_test = np.array(self.labels)[classes_test]

        print(self.x_train.shape[0], 'train samples')
        print(self.x_valid.shape[0], 'valid samples')
        print(self.x_test.shape[0], 'test samples')
        print(len(classes), 'total samples')

        # convert class vectors to binary class matrices

        self.y_train = keras.utils.to_categorical(self.classes_train, self.num_classes)
        self.y_valid = keras.utils.to_categorical(self.classes_valid, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.classes_test, self.num_classes)


    def plot_accuracies(self):
        accuracy_training_means = [np.mean(x) for x in np.transpose(self.accuracy_training_array)]
        accuracy_training_std = [np.std(x) for x in np.transpose(self.accuracy_training_array)]
        accuracy_training_sem = np.array(accuracy_training_std) / [np.sqrt(len(self.x_test))]

        accuracy_valid_means = [np.mean(x) for x in np.transpose(self.accuracy_valid_array)]
        accuracy_valid_std = [np.std(x) for x in np.transpose(self.accuracy_valid_array)]
        accuracy_valid_sem = np.array(accuracy_valid_std) / [np.sqrt(len(self.x_test))]

        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # summarize history for accuracy
        plt.figure()
        plt.errorbar(list(range(len(accuracy_training_means))),accuracy_training_means, yerr=accuracy_training_sem)
        plt.errorbar(list(range(len(accuracy_valid_means))), accuracy_valid_means, yerr=accuracy_valid_sem)
        plt.plot(self.max_valid_epochs + np.random.uniform(-0.1,0.1,size = len(self.max_valid_epochs)),self.max_valid_accuracies,'o',markerfacecolor='None')
        #plt.plot(history.history['acc'])
        #plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['best_test','train:'+str(self.x_train.shape[0]), 'test:'+str(self.x_test.shape[0])], loc='upper left')
        plt.savefig(self.results_path + "/plots/" + self.dataset_name + "_" + self.init + "_batch" + str(self.batch_size) + "_nrep"+str(self.nrep)+'_accuracy.png')
        plt.close()

    def plot_losses(self):
        losses_training_means = [np.mean(x) for x in np.transpose(self.losses_training_array)]
        losses_training_std = [np.std(x) for x in np.transpose(self.losses_training_array)]
        losses_training_sem = np.array(losses_training_std) / [np.sqrt(len(self.x_test))]

        losses_valid_means = [np.mean(x) for x in np.transpose(self.losses_valid_array)]
        losses_valid_std = [np.std(x) for x in np.transpose(self.losses_valid_array)]
        losses_valid_sem = np.array(losses_valid_std) / [np.sqrt(len(self.x_test))]

        # summarize history for loss
        plt.figure()

        plt.errorbar(list(range(len(losses_training_means))),losses_training_means, yerr=losses_training_sem)
        plt.errorbar(list(range(len(losses_valid_means))), losses_valid_means, yerr=losses_valid_sem)
        plt.plot(self.min_valid_loss_epochs + np.random.uniform(-0.1,0.1,size = len(self.min_valid_loss_epochs)),self.min_valid_loss,'o',markerfacecolor='None')

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train:'+str(self.x_train.shape[0]), 'test:'+str(self.x_test.shape[0])], loc='upper left')
        plt.savefig(self.results_path + "/plots/" + self.dataset_name + "_" + self.init + "_batch" + str(self.batch_size) + "_nrep"+str(self.nrep)+'_loss.png')
        plt.close()

