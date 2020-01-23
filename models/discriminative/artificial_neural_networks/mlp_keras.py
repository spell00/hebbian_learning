from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import operator
from utils.utils import dict_of_int_highest_elements, plot_evaluation
from models.NeuralNet import NeuralNet


def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    """
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    :param initial_lr:
    :param decay_factor:
    :param step_size:
    :return:
    """

    from keras.callbacks import LearningRateScheduler
    import numpy as np

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)

# TODO 
# TODO : To make it simpler, if the user want all the layers to be the same and wants a large one.
# TODO : It should be easier to make the network with making a huge list of hundreds of numbers.
# TODO : Example: [1024:2;256:4,128:6] would create a network of [1024,1024,256,256,256,256,128,128,128,128,128,128]
import numpy as np


def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    from keras.callbacks import LearningRateScheduler

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)


lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)


class mlp_keras(NeuralNet):
    def __init__(self):
        super().__init__()

        # empty objects
        #self.csv_filename = csv_filename
        self.model = None
        self.epoch = 0
        self.batch_size = None
        self.n_hidden = [None]
        self.init = None
        self.activation = [None]
        self.l1 = [None]
        self.l2 = [None]
        self.loss_diff = {}
        self.accuracy_diff = {}
        self.dropouts = [None]
        self.loss = [None]
        self.optimizer = [None]
        self.metrics = [None]

    def load_model(self):
        print("\n\nLoading model...\n\n")
        name = "/".join([self.models_path, self.dataset_name]) + "_keras_model"

        model_path = name + ".h5"
        try:
            self.model = keras.models.load_model(model_path)
            print("The model was found at " + model_path + "\n\n")
            self.accuracy_training_array = np.load(name + "_accuracy_trainin_array.npy").tolist()
            self.accuracy_valid_array = np.load(name + "_accuraccy_valid_array.npy").tolist()
            self.max_valid_accuracies = np.load(name + "_max_valid_accuracies.npy").tolist()
            self.max_valid_epochs = np.load(name + "_max_valid_epochs.npy").tolist()
            self.min_valid_loss = np.load(name + "_min_valid_loss.npy").tolist()
            self.min_valid_loss_epochs = np.load(name + "_min_valild_loss_epochs.npy").tolist()
            self.losses_training_array = np.load(name + "_losses_training_array.npy").tolist()
            self.losses_valid_array = np.load(name + "_losses_valid_array.npy").tolist()

        except:
            print("There is no model file by that name at \n" + model_path + "\nStarting from start (epoch 1)")

    def save_model(self):
        print("Saving model..")
        name = "/".join([self.models_path, self.dataset_name]) + "_keras_model"
        self.model.save(name + ".h5")

        np.save(name + "_accuracy_trainin_array.npy", self.accuracy_training_array)
        np.save(name + "_accuraccy_valid_array.npy", self.accuracy_valid_array)
        np.save(name + "_max_valid_accuracies.npy", self.max_valid_accuracies)
        np.save(name + "_max_valid_epochs.npy", self.max_valid_epochs)
        np.save(name + "_min_valid_loss.npy", self.min_valid_loss)
        np.save(name + "_min_valild_loss_epochs.npy", self.min_valid_loss_epochs)
        np.save(name + "_losses_training_array.npy", self.losses_training_array)
        np.save(name + "_losses_valid_array.npy", self.losses_valid_array)

    def init_keras_paramters(self,loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"], n_hidden=[1024, 1024, 1024], init="he_uniform",
                             activation=['relu', 'relu', 'relu'], l1=[0.1, 0.1, 0.1], l2=[0, 0, 0],
                             dropouts=[0.5, 0.5, 0.5]):
        self.n_hidden = n_hidden
        self.init = init
        self.activation = activation
        self.l1 = l1
        self.l2 = l2
        self.dropouts = dropouts
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def build_keras_net(self):
        n_stuff = sum([len(self.n_hidden), len(self.init), len(self.activation), len(self.l1), len(self.l2), len(self.dropouts)])
        if n_stuff == 0:
            print("The model's parameters have not been initialized with: nn.mlp_keras.init_keras_paramters(args)")
            exit()
        elif n_stuff < (len(self.n_hidden)*6):
            print("The model is not well initialized. There is a problem with the pipeline...")
            exit()
        print("Building the network...")
        self.model = Sequential()
        self.model.add(Dense(self.n_hidden[0], activation=self.activation[0], kernel_initializer=self.init,
                             kernel_regularizer=keras.regularizers.l1_l2(self.l1[0], self.l2[0]),
                             input_shape=[self.x_train.shape[1]]))
        self.model.add(Dropout(self.dropouts[0]))
        self.model.add(BatchNormalization())

        for n in range(len(self.n_hidden[1:-1])):
            self.model.add(Dense(self.n_hidden[0], activation=self.activation[n], kernel_initializer=self.init,
                                 kernel_regularizer=keras.regularizers.l1_l2(self.l1[n], self.l2[n])))
            self.model.add(Dropout(self.dropouts[n]))
            self.model.add(BatchNormalization())
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.csv_path = "/".join([self.results_path,self.destination_folder,self.csv_filename])

    def evaluate(self):
        print("Evaluating")
        return self.model.evaluate(x=self.x_test, y=self.y_test)

    def evaluate_individual_arrays(self,top_index=[]):
        print("Evaluating individually")
        total = len(top_index)
        self.loss_diff_indiv = {}
        for i,name in enumerate(top_index):
            self.loss_diff_indiv[name] = []
            #self.accuracy_diff_indiv[name] = []
            if i % 10 == 0:
                print("Evaluation progress {:2.2%}".format(i / total), end="\r")

            tmp_x = np.array(self.x_test)
            tmp_x[:, i] = [0] * len(list(self.classes_test))
            for j in range(len(tmp_x)):
                tmp = self.model.evaluate(x=np.array([tmp_x[j]]), y=np.array([self.y_test[j]]), verbose=0)
                self.log_loss_diff_indiv[name] += [np.log(tmp[0])]
                #self.accuracy_diff_indiv[name] += [tmp[1]]

    def evaluate_individual(self):
        print("Evaluating individually")
        total = len(self.meta_df.index)
        for i,name in enumerate(self.meta_df.index):
            self.loss_diff[name] = []
            self.accuracy_diff[name] = []
            if i % 100 == 0:
                print("Evaluation progress {:2.2%}".format(i / total), end="\r")

            tmp_x = np.array(self.x_test)
            tmp_x[:, i] = [0] * len(list(self.classes_test))
            tmp = self.model.evaluate(x=tmp_x, y=self.y_test,verbose=0)
            self.loss_diff[name] = tmp[0]
            self.accuracy_diff[name] = tmp[1]


    def train(self, lr_sched, nrep=1, n_epochs=10, batch_size=128):
        """

        :param dataset_names:
        :param destination_folder:
        :param results_folder:
        :param n_hidden:
        :param batch_size:
        :param epochs:
        :param nrep:
        :param init:
        :param activation:
        :param l1:
        :param l2:
        :return:

        example:
        from utils import get_example_datasets
        home_path="/home/simon/"
        geo_ids = ["GSE22845","GSE12417"]
        g = get_example_datasets(home_path=home_path, is_load_from_disk=True)
        g.getGEO(geo_ids, is_load_from_disk=True)
        g.merge_datasets(fill_missing=True)
        results_folder = "results"
        mlp = nn(home_path=home_path,results_folder=results_folder)
        mlp.import_dataframe(g.meta_df)
        #mlp.load_dataframe() # TODO

        n_hidden=1024
        batch_size = 128
        epochs = 100
        nrep = 10
        init = "he_uniform"
        activation='relu'
        l1=0.0001
        l2=0.0001


        mlp.keras_mlp()

        """
        # the data, shuffled and split between train and valid sets
        self.batch_size = batch_size
        self.nrep = nrep
        self.n_epochs = n_epochs
        csv_log = self.csv_logger_path + "/" + self.csv_filename + "_logger_mlp.csv"
        if self.model is None: # nrep = 1 otherwise it would be messed up
            self.build_keras_net()
        for rep in range(nrep):
            self.set_data()
            self.csv_logger = keras.callbacks.CSVLogger(csv_log)
            self.model.summary()
            history = self.model.fit(self.x_train, self.y_train, batch_size=int(self.batch_size),
                                     epochs=int(n_epochs), verbose=1, validation_data=(self.x_test, self.y_test),
                                     callbacks=[self.csv_logger,lr_sched])
            max_epoch, max_valid = max(enumerate(history.history['val_acc']), key=operator.itemgetter(1))
            min_loss_epoch, min_loss = min(enumerate(history.history['val_loss']), key=operator.itemgetter(1))
            self.epoch += n_epochs
            self.accuracy_training_array += [history.history['acc']]
            self.accuracy_valid_array += [history.history['val_acc']]
            self.max_valid_accuracies += [max_valid]
            self.max_valid_epochs += [max_epoch]
            self.min_valid_loss += [min_loss]
            self.min_valid_loss_epochs += [min_loss_epoch]
            self.losses_training_array += [history.history['loss']]
            self.losses_valid_array += [history.history['val_loss']]
            self.save_model()
            self.plot_losses()
            self.plot_accuracies()


def test_mlp():
    from geoParser import geoParser
    import torch
    #from dimension_reduction.pca import pca2d
    geo_ids = ["GSE33000"]
    home_path = "/Users/simonpelletier/"
    results_folder = "results"
    data_folder = "data"
    destination_folder = "dementia"
    dataframes_destination = "dataframes"
    meta_destination_folder = "meta_pandas_dataframes"
    initial_lr = 1e-3
    init = "he_uniform"
    n_epochs = 100
    batch_size = 8
    nrep = 1
    silent = 0
    csv_filename = "csv_loggers"
    reps = 1
    hidden_size = 128

    if not torch.cuda.is_available():
        # Install the plaidml backend
        print('Using plaidml')
        import plaidml.keras as plaid_keras
        plaid_keras.install_backend()

    g = geoParser(home_path=home_path, results_folder=results_folder, data_folder=data_folder,
                  destination_folder=destination_folder, dataframes_folder=dataframes_destination,
                  geo_ids=geo_ids, initial_lr=initial_lr, init=init, n_epochs=n_epochs,
                  batch_size=batch_size, hidden_size=hidden_size, silent=silent)

    g.getGEO(is_load_from_disk=True)

    g.merge_datasets(fill_missing=True, is_load_from_disk=True, meta_destination_folder=meta_destination_folder)
    #pca2d(g.meta_df)


    g = geoParser(home_path=home_path,geo_ids=geo_ids,destination_folder=destination_folder,initial_lr=initial_lr, init=init, n_epochs=n_epochs,
                    batch_size=batch_size, hidden_size=hidden_size, silent=silent)
    g.getGEO(geo_ids,is_load_from_disk=True)
    g.merge_datasets()

    mlp = mlp_keras()

    mlp.import_dataframe(g.meta_df)
    mlp.set_configs(home_path=g.home_path, results_folder=g.results_folder, data_folder=g.data_folder,
                    destination_folder=g.destination_folder, dataset_name=g.meta_filename,
                    meta_destination_folder=g.meta_destination_folder, csv_filename=csv_filename)
    mlp.import_dataframe(g.meta_df)
    mlp.init_keras_paramters()
    mlp.set_data()
    for r in range(reps):
        if nrep == 1:
            mlp.load_model()
        print(str(r) + "/" + str(reps))
        mlp.train(batch_size=batch_size, n_epochs=n_epochs, nrep=nrep,lr_sched=lr_sched)
    mlp.evaluate_individual()

    top_loss_diff = dict_of_int_highest_elements(mlp.loss_diff,20) # highest are the most affected
    top_index = top_loss_diff.keys()

    mlp.evaluate_individual_arrays(top_index)
    file_name =  mlp.results_path + "/" + mlp.destination_folder + "/plots/" + mlp.dataset_name + "_" + mlp.init + "_batch" + str(mlp.batch_size) + "_nrep" + str(mlp.nrep)
    plot_evaluation(mlp.log_loss_diff_indiv, file_name + "_accuracy_diff.png",20)

#test_mlp()