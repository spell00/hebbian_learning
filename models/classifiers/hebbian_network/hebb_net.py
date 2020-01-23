from torch.autograd import Variable
import torch.nn.functional as F
from models.classifiers.hebbian_network.hebb_functions import \
    hebb_values_transform_conv, add_conv_units,hebbian_pruning_conv, hebbian_pruning, add_neurons
import numpy as np
import torch
import torch.nn as nn
from models.nn_classifier import nn_classifier


class hebb_net(nn_classifier):
    def __init__(self,
                 Ns=[1024,1024],
                 n_remove=[10, 10],
                 new_ns=[256, 256],
                 hebb_rates_multiplier=[0.0001, 0.0001],
                 gt=[-1000, -1000],
                 gt_convs=None,
                 planes=[16,32,64,128,256,512],
                 kernels=[3,3,3,3,3,3],
                 pooling_layers=[1,1,1,1,1,1],
                 hebb_rates_conv_multiplier=[0.5,0.5,0.5,0.5,0.5,0.5],
                 new_ns_convs=[16,32,64,128,256,512],
                 optim_type="adam",init_function="glorot",
                 nclasses=3, lr=1e-3, clampmax=1000, padding_pooling = 1, padding_no_pooling = 1, n_channels=1,
                 hebb_max_value=1000, dropout = 0.5, how_much_more=0.0, hyper_count = 3,
                 keep_grad=True, pruning=True, wn=False, bn=True, relu=True, hb=True, is_conv=False, schedule_value=0):
        super()#.__init__(nn_classifier)

        #######################       Values from the constructor        #####################

        # Booleans
        if gt_convs is None:
            gt_convs = [-10, -10, -10, -10, -10, -10]
        self.conv = is_conv
        self.wn = wn
        self.bn = bn
        self.relu = relu
        self.hb = hb
        self.keep_grad = keep_grad
        self.pruning = pruning

        # Integers
        self.Ns = Ns
        self.n_channels = n_channels
        self.kernels = kernels
        self.nclasses = nclasses
        self.clampmax = clampmax
        self.hyper_count = hyper_count
        self.schedule_value = schedule_value


        # Floats
        self.dropout = dropout
        self.lr = lr
        self.how_much_more = how_much_more

        # Lists
        self.planes = [n_channels].extend(planes)
        self.padding_pooling = padding_pooling
        self.padding_no_pooling = padding_no_pooling
        self.pooling_layers = pooling_layers
        self.hebb_max_value = hebb_max_value
        self.n_remove = n_remove
        self.hebb_rates_conv_multiplier = hebb_rates_conv_multiplier
        self.hebb_rates_multiplier = hebb_rates_multiplier
        self.new_ns = new_ns
        self.new_ns_convs = new_ns_convs
        self.gt = gt
        self.gt_convs = gt_convs
        self.hebbs = [[]] * len(self.Ns)
        self.hebbs_conv = [[]] * len(kernels)

        # Strings ( call for pytorch's initialization and optimization functions )
        self.set_init(init_function)
        self.set_optim(optim_type)

        ##################    Empty variables and other variables independent from constructor     #####################

        # Booleans
        self.descending = True
        self.is_conv = is_conv

        # Models
        self.best_model = None
        self.last_model = None

        # Integers
        self.best_loss = 100000
        self.best_acc = 0
        self.last_epoch = 0
        self.best_epoch = 0
        self.count = 0

        # Lists
        self.convs = nn.ModuleList()
        self.convs_bn = nn.ModuleList()
        self.kernels_pooling = []
        self.running_losses = []
        self.running_accuracies = []
        self.n_neurons = []

        self.running_train_losses = []
        self.running_valid_losses = []
        self.running_test_losses = []
        self.running_train_accuracies = []
        self.running_valid_accuracies = []
        self.running_test_accuracies = []

        # Set to 0 at first; will be determinated at first iteration
        self.hebb_rates = [0, 0]

        if is_conv:
            self.hebb_rates_conv = [0, 0, 0, 0, 0, 0]
            self.n_convs_layers = []
            for i in range(len(kernels)):
                if(self.pooling_layers[i] == 1):
                    conv2d = nn.Conv2d(planes[i], planes[i + 1], kernels[i], padding=padding_pooling)
                    self.kernels_pooling.append(kernels[i])
                elif (self.pooling_layers[i] == 0):
                    conv2d = nn.Conv2d(planes[i], planes[i + 1], kernels[i], padding=padding_no_pooling)
                if (wn == True): conv2d = nn.utils.weight_norm(conv2d)
                self.convs += [conv2d]
                self.convs_bn += [nn.BatchNorm2d(planes[i + 1])]
                self.hebbs_conv[i] = Variable(torch.zeros(planes[i+1]))

            self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(planes[-1], Ns[0])

        self.fc1.weight = self.init_function(self.fc1.weight)
        self.fc1.bias.data[:] = 0.0
        if wn : self.fc1 = nn.utils.weight_norm(nn.Linear(planes[-1], Ns[0]))
        self.fcs = nn.ModuleList()

        if bn :
            self.bns = nn.ModuleList()
            self.bns += [nn.BatchNorm1d(Ns[0])]
        for i in range(len(Ns)-1):
            self.fcs += [nn.Linear(Ns[i], Ns[i+1])]
            self.fcs[i].weight = self.init_function(self.fcs[i].weight)
            if wn : self.fcs[i] = nn.utils.weight_norm(self.fcs[i])
            self.fcs[i].bias.data[:] = 0.0
            if hb : self.hebbs[i] = Variable(torch.zeros(Ns[i]))
            if bn : self.bns += [nn.BatchNorm1d(Ns[i])]
        self.hebbs[-1] = Variable(torch.zeros(Ns[-1]))
        if bn : self.bns += [nn.BatchNorm1d(Ns[-1])]
        self.fclast = nn.Linear(Ns[-1], nclasses)
        self.fclast.weight = self.init_function(self.fclast.weight)
        if (wn == True): self.fclast = nn.utils.weight_norm(self.fclast)
        self.fclast.bias.data[:] = 0.0
        self.optimizer = self.optimization(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        if (torch.cuda.is_available()):
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.backends.cudnn.benchmark = True
        self.last_epoch += 1
        if self.is_conv:
            for i in range(len(self.convs)):
                if self.wn: x = nn.utils.weight_norm(x)
                x = self.convs[i](x)
                if self.pooling_layers[i] == 1:
                    if self.relu : x = F.relu(self.pool(x))
                elif self.pooling_layers[i] == 0:
                    if self.relu : x = F.relu(x)
                if self.bn:
                    self.convs_bn[i] = nn.BatchNorm2d(self.convs[i].out_channels)
                    x = self.convs_bn[i](x)
                x = F.dropout2d(x,training=self.trainable)
                hebbs_conv = self.hebbs_conv[i].data.copy_(self.hebbs_conv[i].data)
                hebbs_conv = Variable(hebbs_conv)
                val_to_add = torch.sum(hebb_values_transform_conv(x, -self.hebb_rates_conv[i]),dim=0)/(x.shape[1]*x.shape[2]*x.shape[3])
                if(torch.cuda.is_available()):
                    val_to_add = val_to_add.cuda()
                    hebbs_conv = hebbs_conv.cuda()
                self.hebbs_conv[i] = torch.add(hebbs_conv, val_to_add)
                self.hebbs_conv[i] = torch.clamp(self.hebbs_conv[i], max=self.clampmax)
            x = self.fc1(x.squeeze())

        x = F.relu(x)
        if self.bn   :
            self.bns[0] = nn.BatchNorm1d(self.fc1.out_features)
            x = self.bns[0](x)
        x = F.dropout(x, training=self.trainable)
        if self.wn   : x = nn.utils.weight_norm(x)
        hebb = Variable(self.hebbs[0].data.copy_(self.hebbs[0].data))
        val_to_add = torch.sum(self.hebb_values_transform(x, -self.hebb_rates[0]), dim=0)
        if (torch.cuda.is_available()):
            val_to_add = val_to_add.cuda()
            hebb = hebb.cuda()

        self.hebbs[0] = torch.add(hebb, val_to_add)
        self.hebbs[0] = torch.clamp(self.hebbs[0], max=self.clampmax)
        for i in range(len(self.fcs)):
            if self.wn: x = nn.utils.weight_norm(self.fcs[i](x))
            x = self.fcs[i](x)
            x = F.relu(x)
            if self.bn:
                self.bns[i+1] = nn.BatchNorm1d(self.fcs[i].out_features)
                x = self.bns[i+1](x)
            x = F.dropout(x, training=self.trainable)
            hebb = Variable(self.hebbs[i + 1].data.copy_(self.hebbs[i + 1].data))
            val_to_add = torch.sum(self.hebb_values_transform(x, -self.hebb_rates[i+1]), dim=0)
            if(torch.cuda.is_available()):
                val_to_add = val_to_add.cuda()
                hebb = hebb.cuda()

            self.hebbs[i+1] = torch.add(hebb, val_to_add)
            self.hebbs[i+1] = torch.clamp(self.hebbs[i+1], max=self.clampmax)

        x = self.fclast(x)
        return x

    def training(self, files_path='/workspace/catsanddogs/'):

        for epoch in range(100):  # loop over the dataset multiple times
            running_loss = 0.0

            for i, data in enumerate(self.trainloader, 0):
                if (i % int(np.floor(len(self.trainloader) / 10)) == 0):
                    print(i, "/", str(len(self.trainloader)))
                    # for j in range(len(self.hebbs_conv)):
                    #    print("conv max: " + str(int(torch.max(self.hebbs_conv[j]))) + ", conv min: " + str(int(torch.min(self.hebbs_conv[j]))))
                    # for j in range(len(self.hebbs)):
                    #    print("fc max: " + str(int(torch.max(self.hebbs[j]))) + ", fc min: " + str(int(torch.min(self.hebbs[j]))))
                inputs, labels = data
                if (torch.cuda.is_available()):
                    inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)
                self.optimizer.zero_grad()
                outputs = self.forward(inputs, self.hebb_rates_conv, self.hebb_rates, trainable=True)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.running_loss += loss.data[0]

                if i == len(self.trainloader) - 1:  # print every batch
                    self.train_loss, self.train_accuracy = self.validate(loader=self.trainloader, loader_name="training")
                    if self.validloader is not None:
                        self.valid_loss, self.valid_accuracy = self.validate(loader=self.validloader, loader_name="validation")
                    self.test_loss, self.test_accuracy = self.validate(loader=self.testloader, loader_name="test")
                    print("running loss: ", self.run)

                    if (epoch > 0):
                        previous_loss = self.running_loss
                        self.previous_acc = self.valid_accuracy
                        print("previous accuracy: ", self.previous_acc)

        print('Finished Training')

    def compute_hebb(self):
        hebb_rates = np.multiply((self.train_accuracy - self.test_accuracy), self.hebb_rates_multiplier)
        hebb_rates_conv = np.multiply((self.train_accuracy - self.test_accuracy), self.hebb_rates_conv_multiplier)
        print("Hebb rates:", hebb_rates)
        if (self.epoch == 0):
            pass
        elif (self.valid_accuracy > self.previous_acc):
            if (self.descending == True):
                self.count += 1
                print("Changing direction: doing better")
            descending = False
        else:
            if (self.descending == False):
                print("Changing direction: doing worst")
            descending = True

        if (self.valid_accuracy > self.best_acc):
            print("Better accuracy!")
            if (self.valid_accuracy > (self.best_acc + self.how_much_more)):
                print("Count reset to 0")
                count = 0
                self.best_acc = self.valid_accuracy
            else:
                print("Improvement not big enough. Count still going up")

            self.save_model(isBest=True)
            np.save("../results/catsanddogs/nets/net_" + self.filename, self.best_model)
        if (self.count == self.hyper_count):
            print("new neurons")

            # Make in-class function for these

            self = add_conv_units(self, new_conv_channels=self.new_ns_convs, keep_grad=True, init="he")
            self = add_neurons(self, self.new_ns, True, 'he')
            count = 0
        elif (self.pruning == True):

            self = hebbian_pruning_conv(self, self.gt_convs)
            self = hebbian_pruning(self, self.gt)
        print("count: ", count)

        self.running_losses.append(self.running_loss)

        running_loss = 0.0

        lens = [len(x) for x in self.hebbs]
        lens_c = [len(x) for x in self.hebbs_conv]
        self.n_neurons.append(lens)
        self.n_convs_layers.append(lens_c)
        self.optimizer = self.optimization(self.parameters(), lr=self.lr)
        print("n_fc_neurons: ", self.n_neurons[-1])
        print("n_convs_layers: ", self.n_convs_layers[-1])
        print("min_conv_layers: ", [int(torch.min(h).data) for h in self.hebbs_conv])
        print("max_conv_layers: ", [int(torch.max(h).data) for h in self.hebbs_conv])
        print("min_fc_neurons: ", [int(torch.min(h).data) for h in self.hebbs])
        print("max_fc_neurons: ", [int(torch.max(h).data) for h in self.hebbs])

        self.running_train_accuracies.append(self.train_accuracy)
        self.running_valid_accuracies.append(self.valid_accuracy)
        self.running_test_accuracies.append(self.test_accuracy)
        self.running_train_losses.append(self.loss_train / len(self.trainloader))
        self.running_valid_losses.append(self.loss_valid / len(self.validloader))
        self.running_test_losses.append(self.loss_test / len(self.testloader))
        run_values = [self.running_train_accuracies,
                      self.running_valid_accuracies,
                      self.running_test_accuracies,
                      self.running_train_losses,
                      self.running_valid_losses,
                      self.running_test_losses]
        np.save("../results/catsanddogs/nets/net_values" + self.filename, run_values)

    def validate(self, loader, loader_name="training"):
        correct = 0
        total = 0
        loss_train = 0.0

        for i, data in enumerate(loader, 0):
           # This will speed this part and should be close from true
           if (i % 10 == 0):
               images, labels = data
               if (torch.cuda.is_available()):
                   images, labels = images.cuda(), labels.cuda()
               outputs = self.forward(Variable(images), trainable=False)
               _, predicted = torch.max(outputs.data, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum()
               loss_train += self.criterion(outputs, Variable(labels)).data[0]
        loss_train_ratio = loss_train / total
        print('[%d, %5d] Loss: %.3f' % (self.last_epoch + 1, i, ))
        # print("Loss:", running_loss)

        train_accuracy = correct / total
        print(loader_name + ' accuracy %d %%' % (100 * train_accuracy))

        return loss_train_ratio, 100*train_accuracy

    def prepare_data(self, x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size=32, files_path=''):
        train_set = torch.utils.data.TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        test_set = torch.utils.data.TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
        self.trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        self.testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
        if x_valid is not None:
            valid_set = torch.utils.data.TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))
            self.validloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            self.validloader = None

    def save_model(self,isBest):
        if isBest:
            pass
        else:
            pass

    def load_model(self):
        pass

    def set_init(self, init_function="glorot"):
        he_choices = ["he","He","Kaiming","kaiming","Kaiming_uniform","kaiming_uniform","kaiming uniform","Kaiming uniform","Kaiming Uniform","Kaiming_Uniform"]
        glorot_choices = ["glorot", "Glorot", "Xavier", "xavier", "xavier_uniform", "xavier uniform", "Xavier_uniform","Xavier_Uniform","Xavier uniform", "Xavier_uniform"]
        if init_function in he_choices:
            flag = False
            if init_function == "he" or init_function == "He" or init_function == "Kaiming" or init_function == "kaiming":
                choice = input("He (kaiming) initialization can be uniform or normal. Which one do you want? [u/n] (default is uniform [u] ) ")
                if choice != "" and choice != "u":
                    self.init_function = torch.nn.init.kaiming_normal_
                    flag = True
            if not flag:
                self.init_function = torch.nn.init.kaiming_uniform_
        if init_function in glorot_choices:
            flag = False
            if init_function == "glorot" or init_function == "Glorot" or init_function == "Xavier" or init_function == "xavier":
                choice = input("He (kaiming) initialization can be uniform or normal. Which one do you want?" \
                               " [u/n] (default is uniform [u] ) ")
                if choice != "" and choice != "u":
                    self.init_function = torch.nn.init.xavier_normal_
                    flag = True
            if not flag:
                self.init_function = torch.nn.init.xavier_normal_
        elif init_function == "uniform" or init_function == "Uniform":
            self.init_function = torch.nn.init.uniform_
        elif init_function == "constant" or init_function == "Constant":
            self.init_function = torch.nn.init.constant_
        elif init_function == "eye" or init_function == "Eye":
            self.init_function = torch.nn.init.eye_
        elif init_function == "dirac" or init_function == "Dirac":
            self.init_function = torch.nn.init.orthogonal_
        elif init_function == "orthogonal" or init_function == "Orthogonal":
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


def tests_hebb_net(ratio_valid=0,ratio_test=20):
    """

    :param ratio_valid:
    :param ratio_test:
    :return:



    """
    x_train = None
    y_train = None
    x_valid = None
    y_valid = None
    x_test = None
    y_test = None
    batch_size = 32
    files_path = '/workspace/catsanddogs/'

    from geoParser import geoParser
    geo_ids = ["GSE22845","GSE12417"]
    nvidia = "yes"
    home_path = "/home/simon/"
    results_folder = "results"
    data_folder = "data"
    destination_folder = "hebbian_learning_ann"
    dataframes_destination = "dataframes"
    meta_destination_folder = "meta_pandas_dataframes"
    initial_lr = 1e-4
    init = "he_uniform"
    n_epochs = 100
    batch_size = 8
    hidden_size = 128
    nrep = 10
    activation = "relu"
    n_hidden = 128
    l1 = 1e-4
    l2 = 1e-4
    silent = 0

    reps = 1

    g = geoParser(home_path=home_path,
                  results_folder=results_folder,
                  data_folder=data_folder,
                  destination_folder=destination_folder,
                  dataframes_folder=dataframes_destination,
                  geo_ids = geo_ids,
                  initial_lr=initial_lr,
                  init=init,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  hidden_size=hidden_size,
                  silent=silent)



    g.getGEO(is_load_from_disk=True)
    g.merge_datasets(fill_missing=True,
                     is_load_from_disk=True,
                     meta_destination_folder=meta_destination_folder)



    hebb_net = hebb_net()
    hebb_net.load_data()


tests_hebb_net()