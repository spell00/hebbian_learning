from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
from models.discriminative.artificial_neural_networks.hebbian_network.figure_generator import plot_performance
from models.NeuralNet import NeuralNet
from models.discriminative.artificial_neural_networks.hebbian_network.utils import hebb_values_transform_conv, hebb_values_transform
import numpy as np
import matplotlib.pyplot as plt
from models.discriminative.artificial_neural_networks.hebbian_network.utils import indices_h, indices_h_conv
from utils.utils import create_missing_folders
import torchvision as tv


def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

class HebbNet(NeuralNet):
    def __init__(self):
        super().__init__()
        # TODO REORGANIZE
        self.epoch = None
        self.convs = None
        self.convs_bn = None
        self.hebb_values_conv = None
        self.new_ns_convs = None
        self.gt_convs = None
        self.hebbs_conv = None
        self.hebb_rates_conv = None
        self.n_convs_layers = None
        self.padding_pooling = None
        self.padding_no_pooling = None
        self.n_channels = None
        self.conv2d = None
        self.pool = None
        self.valid_indices = None
        # Models
        self.best_model = None
        self.last_model = None

        self.hebb_input_values = None
        self.valid_inputs = None
        self.invalid_indices = None
        self.tanh = None

        # Integers
        self.best_loss = 100000
        self.best_acc = 0
        self.previous_loss = 0
        self.last_epoch = 0
        self.best_epoch = 0
        self.count = 0
        self.batch_size = None
        # Lists
        self.kernels_pooling = []
        self.running_losses = []
        self.running_accuracies = []
        self.n_neurons = [[]]

        self.running_train_losses = []
        self.running_valid_losses = []
        self.running_test_losses = []
        self.running_train_accuracies = []
        self.running_valid_accuracies = []
        self.running_test_accuracies = []

        # Dicts
        self.labels_dict = {"train": [], "valid": [], "valid": []}
        self.accuracies_dict = {"train": [], "valid": [], "valid": []}
        self.losses_dict = {"train": [], "valid": [], "valid": []}

        # Set to 0 at first; will be determinated at first iteration
        self.relu = None
        self.hb = None
        self.keep_grad = None
        self.is_pruning = None
        self.nclasses = None
        self.clampmax = None
        self.hyper_count = None
        self.schedule_value = None
        self.dropout = None
        self.lr = None
        self.how_much_more = None
        self.hebb_max_value = None
        self.descending = None
        self.is_conv = None
        self.bn = None
        self.wn = None
        self.alive_neurons = None
        self.bn_input = None

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.bns = None
        self.optimizer = None

        self.criterion = None
        self.gt_input = None
        self.input_size = None
        self.fcs = None
        self.num_classes = None
        self.hebb_input_values = None
        self.hebb_rate_input = 0

        self.hebb_values = [0., 0]
        self.hebb_values_neurites = [0., 0.]
        self.hebb_rates = [0., 0.]
        self.hebb_rates_inputs = [0., 0.]
        self.hebb_rates_neurites = [0., 0.]
        self.Ns = [512, 512]
        self.new_ns = [32, 32]
        self.hebb_rates_multiplier = [0., 0.]
        self.hebb_rates_inputs_multiplier = 1
        self.hebb_rates_neurites_multiplier = [0., 0.]
        self.gt = [0., 0.]
        self.gt_neurites = [0., 0.]
        self.gt_convs = [-10, -10]
        self.planes = [16, 32, 64, 128, 256, 512]
        self.kernels = [3, 3, 3, 3, 3, 3]
        self.pooling_layers = [1, 1, 1, 1, 1, 1]
        self.hebb_rates_conv_multiplier = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.new_ns_convs = [16, 32, 64, 128, 256, 512]

    def init_parameters(self, Ns=None, hebb_rates=None, hebb_rates_neurites=None, hebb_rates_multiplier=None,
                        new_ns=None, gt=None, kernels=None, gt_neurites=None, lambd=0.01,
                        optim_type="adam", init="glorot", lr=1e-3, clampmax=1000, gt_input=-100,
                        hebb_rate_input=0., hebb_input_values=0, l2=0., gt_convs=None, new_ns_convs=None,
                        padding_pooling=1, padding_no_pooling=1, n_channels=1, hebb_max_value=1000, dropout=0.5, l1=0.,
                        how_much_more=0.0, hyper_count=3, keep_grad=True, is_pruning=True, wn=False, bn=True, relu=True,
                        hb=True, is_conv=False, schedule_value=0, batch_size=None, input_size=None, num_classes=None):

        self.init = init
        # Booleans
        self.n_channels = n_channels
        self.is_conv = is_conv
        self.gt_input = gt_input
        self.wn = wn
        self.bn = bn
        self.relu = relu
        self.lambd = lambd
        self.hb = hb
        self.keep_grad = keep_grad
        self.is_pruning = is_pruning
        self.input_size = input_size * n_channels
        self.valid_bool = list(range(self.input_size))
        self.alive_inputs = list(range(self.input_size))
        self.previous_valid_len = self.input_size
        self.num_classes = num_classes

        # Integers
        self.clampmax = clampmax
        self.hyper_count = hyper_count
        self.schedule_value = schedule_value
        self.l1 = l1
        self.l2 = l2
        self.batch_size = batch_size
        self.hebb_rate_input = hebb_rate_input
        self.hebb_input_values = hebb_input_values
        # Floats
        self.dropout = dropout
        self.lr = lr
        self.how_much_more = how_much_more

        self.hebb_max_value = hebb_max_value
        if Ns is not None:
            self.Ns = Ns
        self.n_neurites = [[] for _ in self.Ns]
        #self.n_neurites[0] += [self.input_size * self.Ns[0]]
        #for i in range(len(self.Ns)-1):
        #    self.n_neurites[i] += [self.Ns[i] * self.Ns[i+1]]
        if hebb_rates_multiplier is not None:
            self.hebb_rates_multiplier = hebb_rates_multiplier
        if hebb_rates_multiplier is not None:
            self.hebb_rates_multiplier = hebb_rates_multiplier
        if hebb_rates_neurites is not None:
            self.hebb_rates_neurites = hebb_rates_neurites
        if hebb_rates is not None:
            self.hebb_rates = hebb_rates
        if new_ns is not None:
            self.new_ns = new_ns
        if gt is not None:
            self.gt = gt
        if gt_neurites is not None:
            self.gt_neurites = gt_neurites
        if gt_convs is not None:
            self.gt_convs = gt_convs
        if kernels is not None:
            self.gt_convs = gt_convs

        self.fcs = nn.ModuleList()
        self.h_encods = nn.ModuleList()
        self.h_decods = nn.ModuleList()
        self.h_encod1 = None
        self.h_decod1 = None

        # Strings ( call for pytorch's initialization and optimization functions )
        self.set_init(init)
        self.set_optim(optim_type)

        ##################    Empty variables and other variables independent from constructor     #####################

        # Booleans
        self.descending = True
        self.is_conv = is_conv
        self.bn = bn
        self.wn = wn

        if is_conv:
            self.convs = nn.ModuleList()
            self.convs_bn = nn.ModuleList()
            self.hebb_values_conv = [[]] * len(kernels)
            self.new_ns_convs = new_ns_convs
            self.gt_convs = gt_convs
            self.hebb_rates_conv = [0, 0, 0, 0, 0, 0]
            self.n_convs_layers = []
            self.padding_pooling = padding_pooling
            self.padding_no_pooling = padding_no_pooling
            self.n_channels = n_channels

        self.criterion = nn.CrossEntropyLoss()
        list1 = [self.input_size] + self.Ns
        self.list1 = list1
        self.hebb_values_neurites = [torch.zeros(list1[i+1], list1[i]) for i in range(len(self.Ns))]
        self.original_num_neurites = [int(x.shape[0] * int(x.shape[1])) for x in self.hebb_values_neurites]

        self.hebb_values = [Variable(torch.Tensor([0] * n)) for n in self.Ns]
        self.n_neurons = [[] for _ in self.Ns]

        lenlen = {
                  "gt"                      : len(self.gt),
                  "hebb_rates_multiplier"   : len(self.hebb_rates_multiplier),
                  "hebb_rates_neurites"     : len(self.hebb_rates_neurites),
                  "hebb_rates"              : len(self.hebb_rates),
                  "new_ns"                  : len(self.new_ns),
                  "gt_neurites"             : len(self.gt_neurites)
                 }
        try:
            assert len(set(lenlen.values())) == 1
        except:
            print(lenlen)
            print("All arguments doesnt have the same lenght. ")
            set_lens = set(lenlen.values())
            for l in set_lens:
                names = print([key for key in lenlen.keys() if lenlen[key] == l], "have lenght", l)
            exit()

    def init_parameters_conv(self, hebb_rates_conv_multiplier, gt_convs, new_ns_convs, planes, kernels,
                             n_channels, padding_pooling, padding_no_pooling, pooling_layers):
        self.n_channels = n_channels
        self.planes = [self.n_channels].extend(planes)
        self.kernels = kernels
        self.padding_pooling = padding_pooling
        self.padding_no_pooling = padding_no_pooling
        self.pooling_layers = pooling_layers
        self.hebb_rates_conv_multiplier = hebb_rates_conv_multiplier
        self.gt_convs = gt_convs
        self.new_ns_convs = new_ns_convs

    def set_layers(self):
        conv2d = None
        self.hebb_input_values = Variable(torch.zeros(self.input_size))
        self.valid_inputs = [True] * self.input_size
        self.invalid_indices = []
        self.tanh = torch.tanh
        if torch.cuda.is_available():
            self.hebb_input_values.cuda()
        if self.bn:
            self.bn_input = nn.BatchNorm1d(self.input_size)
        if self.bn:
            self.bns = nn.ModuleList()
        if self.is_conv:
            for i in range(len(self.kernels)):
                if self.pooling_layers[i] == 1:
                    conv2d = nn.Conv2d(self.planes[i], self.planes[i+1], self.kernels[i], padding=self.padding_pooling)
                    self.kernels_pooling.append(self.kernels[i])
                elif self.pooling_layers[i] == 0:
                    conv2d = nn.Conv2d(self.planes[i], self.planes[i + 1], self.kernels[i],
                                       padding=self.padding_no_pooling)
                if self.self.wn:
                    self.conv2d = nn.utils.weight_norm(conv2d)
                self.convs += [self.conv2d]
                if self.bn:
                    self.convs_bn += [nn.BatchNorm2d(self.planes[i + 1])]
                self.hebb_values_conv[i] = Variable(torch.zeros(self.planes[i + 1]))

            self.pool = nn.MaxPool2d(2, 2)
            self.fcs[0] = nn.Linear(self.planes[-1], self.Ns[0])
            self.fcs[0].weight.data = Variable(self.init_function(self.fcs[i].weight), requires_grad=True)
            self.fcs[0].bias.data = Variable(torch.zeros(self.fcs[i].bias.data.shape))

            self.bns += [nn.BatchNorm1d(self.Ns[0])]
            if self.wn:
                self.fcs[i] = nn.utils.weight_norm(nn.Linear(self.planes[-1], self.Ns[0]))
        else:
            self.fcs[0] = nn.Linear(self.input_size, self.Ns[0])
            self.fcs[0].weight = nn.Parameter(Variable(self.init_function(torch.zeros(self.fcs[0].weight.data.shape))))
            self.fcs[0].weight.grad = nn.Parameter(Variable(torch.zeros(self.fcs[0].weight.data.shape)))
            self.fcs[0].bias.data[:] = 0.0
            self.fcs[0].bias.grad = nn.Parameter(Variable(torch.zeros(self.fcs[0].bias.data.shape)))
            self.bns[0] = nn.BatchNorm1d(self.Ns[0])

        for i in range(1, len(self.Ns)):
            self.fcs += [nn.Linear(self.Ns[i-1], self.Ns[i])]
            self.fcs[i].weight = nn.Parameter(Variable(self.init_function(torch.zeros(self.fcs[i].weight.data.shape))))
            self.fcs[i].weight.grad = nn.Parameter(Variable(torch.zeros(self.Ns[i], self.Ns[i-1])))
            self.fcs[i].bias.data[:] = 0.0
            self.fcs[i].bias.grad = nn.Parameter(Variable(torch.zeros(self.fcs[i].bias.data.shape)))
            if self.bn:
                self.bns += [nn.BatchNorm1d(self.Ns[i])]
        self.fcs += [nn.Linear(self.Ns[-1], self.num_classes)]
        self.fcs[-1].weight = nn.Parameter(Variable(self.init_function(torch.zeros(self.fcs[-1].weight.data.shape))))
        self.fcs[-1].weight.grad = nn.Parameter(Variable(torch.zeros(self.fcs[-1].weight.data.shape)))
        self.fcs[-1].bias.data[:] = 0.0
        self.fcs[-1].bias.grad = nn.Parameter(Variable(torch.zeros(self.fcs[-1].bias.data.shape)))


    def validate(self, loader, epoch, loader_name="training", display_rate=10, verbose=2):
        labels = None
        correct = 0
        total = 0
        loss_train = 0.0
        i = 0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            tv.utils.save_image(inputs[0][0], "test_hnet_original_image.png")
            inputs = Variable(torch.FloatTensor(inputs.detach().cpu().numpy()))
            try:
                labels = Variable(torch.LongTensor(
                    [x.index(1) for i, x in enumerate(labels.detach().cpu().numpy().tolist())]))
            except:
                pass
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.forward(inputs, trainable=False)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if torch.cuda.is_available():
                correct += np.sum(predicted.cpu().numpy() == labels.cpu().numpy())
            else:
                correct += np.sum(predicted.numpy() == labels.numpy())
            loss_train += self.criterion(outputs, labels).item()
        loss_train_ratio = loss_train / total
        train_accuracy = correct / total
        if epoch % display_rate == 0 and verbose > 0:
            print('\n[%d, %5d] Loss: %.3f' % (self.last_epoch + 1, i, loss_train_ratio))
            print(loader_name + ' accuracy %.3f %%' % (100 * train_accuracy), "\n")

        return loss_train_ratio, 100 * train_accuracy

    def save_model(self, is_best):
        if is_best:
            torch.save(self.state_dict(), './net.pth')
        else:
            pass

    def load_model(self):
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

    def calculate_neurons_usage(self):
        pass

    def calculate_neurites_usage(self):
        pass

    def print_parameters(self):
        print("Optimizer", self.optimizer)
        print("fcs", self.fcs)
        for i, fc in enumerate(self.fcs):
            print("fcs", i, ":", fc)
            print("fcs grad", i, ":", fc.weight.grad.shape)
            print("fcs weight", i, ":", fc.weight.shape)
        print("bns[i]", self.bns[i])
        for i, bn in enumerate(self.bns):
            print("bns", i, ":", bn)

    def run(self, n_epochs=100, hebb_round=10, verbose=3, display_rate=10, display_progress=True):
        self.count = 0
        self.count_down = 0
        self.best_loss = 1000000
        self.descending = False
        self.how_much_more = 0
        self.is_conv = False
        self.is_pruning = True
        loss = None
        inputs = None
        outputs = None
        labels = None
        train_accuracy = None
        valid_accuracy = None
        test_accuracy = None
        train_loss = None
        valid_loss = None
        test_loss = None

        self.optimizer = self.optimization(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    verbose=True, cooldown=0, patience=5)
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            self.epoch = epoch
            self.optimizer = self.optimization(self.parameters(), lr=self.lr)
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                if display_progress and verbose > 0:
                    print("Epoch",epoch,"progress {:2.0%}".format(i / len(self.train_loader)), end="\r")
                inputs, labels = data
                inputs = Variable(torch.FloatTensor(inputs.detach().cpu().numpy()))
                try:
                    labels = Variable(
                        torch.LongTensor([x.index(1) for i, x in enumerate(labels.cpu().numpy().tolist())]))
                except:
                    pass

                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                outputs = self.forward(inputs, trainable=True)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()

                self.optimizer.step()
                running_loss += loss.item()
                if i == len(self.train_loader) - 1:  # print every batch
                    image_path = self.results_path + "/images/examples/" + self.dataset_name + "/"
                    create_missing_folders(image_path)
                    img_path = "_".join(["img", str(self.previous_valid_len), str(self.epoch), ".png"])
                    if self.n_channels == 1:
                        mask = np.reshape(self.valid_bool, newshape=inputs[0][0].shape)  # TODO change hard coding
                        img = inputs[0][0].cpu().numpy() * mask
                        from matplotlib import cm
                        plt.imsave(image_path + img_path, img, cmap=cm.gray)

                    running_loss = running_loss/len(self.train_loader)
                    train_loss, train_accuracy = \
                        self.validate(loader=self.train_loader, epoch=epoch, loader_name="training",
                                      display_rate=display_rate, verbose=verbose)
                    if self.valid_loader is not None:
                        valid_loss, valid_accuracy = \
                            self.validate(loader=self.valid_loader, epoch=epoch, loader_name="validation",
                                          display_rate=display_rate, verbose=verbose)
                    test_loss, test_accuracy = \
                        self.validate(loader=self.test_loader, epoch=epoch,
                                      loader_name="valid", display_rate=display_rate, verbose=verbose)

                    if epoch % display_rate == 0 and verbose > 0:
                        print("\nEPOCH ", epoch, "\n")
                        print("running loss: ", running_loss)
            alive_neurites1 = self.hebb_values_neurites[0] > self.gt_neurites[0]
            if torch.cuda.is_available():
                alive_neurites1 = torch.Tensor(alive_neurites1.data.cpu().numpy()).cuda()
            else:
                alive_neurites1 = torch.Tensor(alive_neurites1.data.numpy())
            self.n_neurites[0] += [int(torch.sum(alive_neurites1))]
            for i in range(len(self.hebb_values_neurites)-1):
                alive_neurites_in = self.hebb_values_neurites[i+1] > self.gt_neurites[i+1]
                if torch.cuda.is_available():
                    alive_neurites_in = torch.Tensor(alive_neurites_in.data.cpu().numpy()).cuda()
                else:
                    alive_neurites_in = torch.Tensor(alive_neurites_in.data.numpy())
                self.n_neurites[i+1] += [int(torch.sum(alive_neurites_in))]

            print()
            if epoch % hebb_round == 0 and epoch != 0:
                self.compute_hebb(running_loss, epoch, verbose)
            if self.is_conv:
                lens_c = [len(x) for x in self.hebb_values_conv]
                self.n_convs_layers.append(lens_c)
            self.scheduler.step(running_loss)
            del loss, inputs, outputs, running_loss

            assert len(self.Ns) == len(self.n_neurons)
            for i in range(len(self.Ns)):
                self.n_neurons[i].append(self.Ns[i])
            input_values = self.hebb_input_values

            self.accuracies_dict["train"].append(train_accuracy)
            self.accuracies_dict["valid"].append(valid_accuracy)
            self.accuracies_dict["valid"].append(test_accuracy)
            self.losses_dict["train"].append(train_loss)
            self.losses_dict["valid"].append(valid_loss)
            self.losses_dict["valid"].append(test_loss)
            filename = "_".join([self.dataset_name, str(self.lr), str(self.dropout), self.init])

            plot_performance(self.accuracies_dict, self.labels_dict, n_list=self.n_neurons,
                             results_path=self.results_path, filename=filename + "_accuracy_neurons.png")
            plot_performance(self.accuracies_dict, self.labels_dict, n_list=self.n_neurites,
                             results_path=self.results_path, filename=filename + "_accuracy_neurites.png")
            plot_performance(self.accuracies_dict, self.labels_dict, n_list=[len(input_values)],
                             results_path=self.results_path, filename=filename + "_accuracy_inputs.png")
            plot_performance(self.losses_dict, self.labels_dict, n_list=self.n_neurons,
                             results_path=self.results_path, filename=filename + "_losses_neurons.png")
            plot_performance(self.losses_dict, self.labels_dict, n_list=self.n_neurites,
                             results_path=self.results_path, filename=filename + "_losses_neurites.png")
            plot_performance(self.losses_dict, self.labels_dict, n_list=[len(input_values)],
                             results_path=self.results_path, filename=filename + "_losses_inputs.png")

            self.optimizer = self.optimization(self.parameters(), lr=self.lr)

            if verbose > 0 and epoch % display_rate is 0:
                print("n_input: ", len(input_values))
                print("hebb_input_values min: ", torch.min(input_values))
                print("hebb_input_values max: ", torch.max(input_values))
            if verbose > 1 and epoch % display_rate is 0:
                for i, n in enumerate(self.Ns):
                    print("#Neurons L", i, ":", n)
                if self.is_conv:
                    print("n_convs_layers: ", self.n_convs_layers[-1])
                    print("min_conv_layers: ", [int(torch.min(h).data) for h in self.hebb_values_conv])
                    print("max_conv_layers: ", [int(torch.max(h).data) for h in self.hebb_values_conv])

            if verbose > 2 and epoch % display_rate is 0:
                print("min_fc_neurons: ", [int(torch.min(h)) for h in self.hebb_values])
                print("max_fc_neurons: ", [int(torch.max(h)) for h in self.hebb_values])


        print('Finished Training')

    def compute_hebb(self, running_loss, epoch, verbose, count_down_limit=10, display_rate=10):
        if verbose > 1:
            print("Hebb rates inputs:", self.hebb_rate_input)
            print("Hebb rates neurons:", self.hebb_rates)
            print("Hebb rates neurites:", self.hebb_rates_neurites)
        if epoch == 0:
            pass
        elif running_loss < self.previous_loss:
            if self.descending:
                self.count += 1
                if verbose > 2:
                    print("Changing direction: doing better")
            self.descending = False
            self.count_down = 0

        else:
            if not self.descending:
                if verbose > 2:
                    print("Changing direction: doing worst")
            else:
                self.count_down += 1
            self.descending = True

        if running_loss < self.best_loss:
            if verbose > 0:
                print("Better loss!")
            if running_loss < (self.best_loss + self.how_much_more):
                if verbose > 2:
                    print("Count reset to 0")
                self.count = 0
                self.best_loss = running_loss
            else:
                if verbose > 0:
                    print("Improvement not big enough. Count still going up")
        print("HYPER COUNT", self.hyper_count)
        if self.count == self.hyper_count or self.count_down == count_down_limit:
            if verbose > 2:
                print("new neurons")
                if self.count == self.hyper_count:
                    print("Reason: Hyper count reached")
                else:
                    print("Reason: Worsening limit reached")
            if self.is_conv:
                self.add_conv_units(new_conv_channels=self.new_ns_convs, keep_grad=True, init="he")
            self.add_neurons(keep_grad=True, init='he')
            self.count = 0
        elif self.is_pruning:
            self.input_pruning()
            if self.is_conv:
                pass
                #self.pruning_conv()
            self.pruning()
        if verbose > 0:
            print("count: ", self.count)
            print("count down: ", self.count_down)

        self.running_losses.append(running_loss)
        if (epoch > 0):
            if epoch % display_rate == 0 and verbose > 1:
                print("previous accuracy: ", self.previous_loss)
                print("running_loss: ", running_loss)
            self.previous_loss = running_loss
            self.previous_acc = self.accuracies_dict["train"]

    def add_hebb_neurites(self, mul, layer):
        hvals_neurites1 = self.hebb_values_neurites[layer]
        hvals_neurites1 = Variable(hvals_neurites1, requires_grad=True)
        hrate_neurites1 = -(int(torch.sum(mul).detach().cpu().numpy()) / (len(mul[0]) * len(mul)))
        self.hebb_rates_neurites[layer] = hrate_neurites1
        if torch.cuda.is_available():
            hvals_neurites1 = hvals_neurites1.cuda()
        matrix_to_add = Variable(hebb_values_transform(mul, hrate_neurites1), requires_grad=True)
        if torch.cuda.is_available():
            matrix_to_add = matrix_to_add.cuda()
            hvals_neurites1 = hvals_neurites1.cuda()

        self.hebb_values_neurites[layer] = torch.add(hvals_neurites1, matrix_to_add)
        self.hebb_values_neurites[layer] = torch.clamp(self.hebb_values_neurites[layer], max=self.clampmax)

    def add_hebb_neurons_input(self, xs):
        x_input = self.bn_input(xs[0])
        hebb_input = Variable(self.hebb_input_values.data.copy_(self.hebb_input_values.data))
        matmul = xs[-1]
        for i in range(len(self.fcs)).__reversed__():
            matmul = torch.matmul(matmul, self.fcs[i].weight)
        mul = torch.mul(x_input, matmul)


        self.hebb_rate_input = -int(torch.mean(mul))
        val_to_add_input = torch.sum(hebb_values_transform(mul, self.hebb_rate_input), dim=0)

        if torch.cuda.is_available():
            val_to_add_input = Variable(val_to_add_input).cuda()
            hebb_input = hebb_input.cuda()
        self.hebb_input_values = self.lambd * torch.add(val_to_add_input, hebb_input)
        self.hebb_input_values = torch.clamp(self.hebb_input_values, min=-1000, max=self.clampmax)

    def add_hebb_neurons_input0(self, x_input, x):
        x_input = self.bn_input(x_input)
        hebb_input = Variable(self.hebb_input_values.data.copy_(self.hebb_input_values.data))

        matmul = torch.matmul(x, self.fcs[0].weight)
        mul = torch.mul(x_input, matmul)
        print("mean mul", torch.mean(mul))
        self.hebb_rate_input = -(int(torch.sum(x_input).detach().cpu().numpy()) / (len(x_input[0]) * len(x_input)))
        val_to_add_input = torch.sum(hebb_values_transform(mul, self.hebb_rate_input), dim=0)

        if torch.cuda.is_available():
            val_to_add_input = Variable(val_to_add_input).cuda()
            hebb_input = hebb_input.cuda()
        self.hebb_input_values = torch.add(val_to_add_input, hebb_input)
        self.hebb_input_values = torch.clamp(self.hebb_input_values, min=-1000, max=self.clampmax)

    def add_hebb_neurons(self, x, i):
        hvals = self.hebb_values[i]
        print(hvals.shape)
        print(x[:5])
        hrate = -(int(torch.sum(x).detach().cpu().numpy()) / (len(x[0]) * len(x)))

        self.hebb_rates[i] = hrate
        vals = hebb_values_transform(x, hrate)
        val_to_add = torch.sum(vals, dim=0)
        if torch.cuda.is_available():
            val_to_add = val_to_add.cuda()
            hvals = hvals.cuda()
        self.hebb_values[i] = torch.add(hvals, val_to_add)
        self.hebb_values[i] = torch.clamp(self.hebb_values[i], max=self.clampmax)



    def pruning(self, min_n_classes=2, minimum_neurons=2):
        for i in range(len(self.gt)):
            if len(self.hebb_values[i]) >= min_n_classes:
                alive_neurons_out = self.hebb_values[i] > float(self.gt[i])
                indices_alive_neurons_out = indices_h(alive_neurons_out)
                self.hebb_values_neurites[i] = self.hebb_values_neurites[i][indices_alive_neurons_out, :]

                w2 = self.fcs[i].weight.data.copy_(self.fcs[i].weight.data).cpu().numpy()
                b2 = self.fcs[i].bias.data.copy_(self.fcs[i].bias.data).cpu().numpy()
                wg2 = self.fcs[i].weight.grad.data.copy_(self.fcs[i].weight.grad.data).cpu().numpy()
                bg2 = self.fcs[i].bias.grad.data.copy_(self.fcs[i].bias.grad.data).cpu().numpy()

                bg2 = bg2[indices_alive_neurons_out]
                b2 = b2[indices_alive_neurons_out]

                wg2 = wg2[indices_alive_neurons_out, :]
                w2 = w2[indices_alive_neurons_out, :]

                if i > 0:
                    alive_neurons_in = torch.Tensor([True if x > float(self.gt[i-1]) else False for x in self.hebb_values[i-1]])
                    indices_alive_neurons_in = indices_h(alive_neurons_in)

                    self.hebb_values_neurites[i] = self.hebb_values_neurites[i][:, indices_alive_neurons_in]
                    wg2 = wg2[:, indices_alive_neurons_in]
                    w2 = w2[:, indices_alive_neurons_in]
                    self.fcs[i].in_features = wg2.shape[1]


                self.Ns[i] = len(b2)
                self.fcs[i].out_features = len(b2)

                b2 = torch.from_numpy(b2)
                bg2 = torch.from_numpy(bg2)
                w2 = torch.from_numpy(w2)
                wg2 = torch.from_numpy(wg2)

                if torch.cuda.is_available():
                    w2 = Variable(w2).cuda()
                    wg2 = Variable(wg2).cuda()
                    b2 = Variable(b2).cuda()
                    bg2 = Variable(bg2).cuda()

                self.fcs[i].weight = nn.Parameter(w2)
                self.fcs[i].weight.grad = nn.Parameter(wg2)
                self.fcs[i].bias = nn.Parameter(b2)
                self.fcs[i].bias.grad = nn.Parameter(bg2)

                #alive_neurites = self.hebb_values_neurites[i] > self.gt_neurites[i]
                #alive_neurites = torch.Tensor(alive_neurites.data.cpu().numpy()).cuda()

                self.hebb_values[i] = self.hebb_values[i][indices_alive_neurons_out]
                #self.fcs[i].weight.data = self.fcs[i].weight.data * alive_neurites
                #self.n_neurites[i] += [int(torch.sum(alive_neurites))]

                if len(indices_alive_neurons_out) < minimum_neurons:
                    indices_alive_neurons_out = indices_h(torch.sort(self.hebb_values[i])[1] < minimum_neurons)
                    print("Minimum neurons on layer ", (i + 1))

        w3 = self.fcs[-1].weight.data.copy_(self.fcs[-1].weight.data).cpu().numpy()
        wg3 = self.fcs[-1].weight.grad.data.copy_(self.fcs[-1].weight.grad.data).cpu().numpy()
        wg3 = wg3[:, indices_alive_neurons_out]
        self.fcs[-1].in_features = len(indices_alive_neurons_out)
        if torch.cuda.is_available():
            self.fcs[-1].weight = nn.Parameter(Variable(torch.from_numpy(w3[:, indices_alive_neurons_out])).cuda())
            self.fcs[-1].weight.grad = nn.Parameter(Variable(torch.from_numpy(wg3)).cuda())
        else:
            self.fcs[-1].weight = nn.Parameter(Variable(torch.from_numpy(w3[:, indices_alive_neurons_out])))
            self.fcs[-1].weight.grad = nn.Parameter(Variable(torch.from_numpy(wg3)))
        if torch.cuda.is_available():
            self = self.cuda()

    def input_pruning(self, min_n_classes=20, minimum_neurons=20):
        """
        :param net:
        :param gt:
        :param min_n_classes:
        :param minimum_neurons:
        :return:
        """
        hebb_input = self.hebb_input_values.data.copy_(self.hebb_input_values.data).cpu().numpy()
        if len(hebb_input) >= min_n_classes:
            to_keep = hebb_input > float(self.gt_input)
            valid_indices = indices_h(to_keep)
            if len(valid_indices) < minimum_neurons:
                # TODO Replace neurons that could not be removed?
                valid_indices = indices_h(torch.sort(hebb_input)[1] < minimum_neurons)
                print("Minimum neurons on layer 1")

            print("previous_valid_len", self.previous_valid_len)
            self.valid_bool = [1. if x in valid_indices else 0. for x in range(self.input_size)]
            self.alive_inputs = [x for x in range(len(hebb_input)) if x in valid_indices]
            alive_inputs = np.array(self.alive_inputs)
            if len(self.alive_inputs) < self.previous_valid_len or self.epoch == 1:
                print("SAVING MASK")
                masks_path = self.results_path + "/images/masks/" + self.dataset_name + "/"
                create_missing_folders(masks_path)
                img_path = "_".join(["alive_inputs", str(len(valid_indices)), str(self.epoch), ".png"])
                if self.n_channels == 1:
                    mask = np.reshape(self.valid_bool, newshape=(28, 28))  # TODO change hard coding
                    plt.imsave(masks_path + img_path, mask)
            self.previous_valid_len = len(valid_indices)

    def add_neurons(self, keep_grad=True, init="he", n_classes=2):
        if keep_grad and init is "he":
            for i in range(len(self.new_ns)):
                if self.new_ns[i] > 0:
                    hebbs = Variable(self.hebb_values[i].data.copy_(self.hebb_values[i].data)).cpu()
                    new_neurons = Variable(torch.zeros(self.new_ns[i]))
                    hebbs = Variable(torch.cat((hebbs, new_neurons)))
                    self.Ns[i] = len(hebbs)
                    hebbs_neurites = Variable(self.hebb_values_neurites[i].data.copy_(self.hebb_values_neurites[i].data)).cpu()
                    new_neurites1 = Variable(torch.zeros(self.new_ns[i], hebbs_neurites.shape[1]))
                    hebbs_neurites = Variable(torch.cat((hebbs_neurites, new_neurites1), dim=0))

                    w2 = self.fcs[i].weight.data.copy_(self.fcs[i].weight.data).cpu()
                    b2 = self.fcs[i].bias.data.copy_(self.fcs[i].bias.data).cpu()
                    wg2 = self.fcs[i].weight.grad.data.copy_(self.fcs[i].weight.grad.data).cpu()
                    bg2 = self.fcs[i].bias.grad.data.copy_(self.fcs[i].bias.grad.data).cpu()
                    new_biases2 = torch.zeros(self.new_ns[i])
                    b2 = torch.cat((b2, new_biases2))
                    bg2 = Variable(torch.cat((bg2, new_biases2)))

                    new_weights1 = torch.zeros([w2.shape[0] + self.new_ns[i], w2.shape[1]])
                    new_weights1 = self.init_function(new_weights1)[0:self.new_ns[i], :]
                    new_weights_grad1 = torch.zeros([w2.shape[0] + self.new_ns[i], w2.shape[1]])[0:self.new_ns[i],:]
                    w2 = torch.cat((w2, new_weights1), dim=0)
                    wg2 = torch.cat((wg2, new_weights_grad1), dim=0)

                    if i > 0:
                        new_neurites2 = Variable(torch.zeros(len(hebbs_neurites), self.new_ns[i-1]))
                        hebbs_neurites = Variable(torch.cat((hebbs_neurites, new_neurites2), dim=1))
                        new_weights2_2 = torch.zeros([w2.shape[0], w2.shape[1] + self.new_ns[i-1]])
                        new_weights2_2 = self.init_function(new_weights2_2)[:, 0:self.new_ns[i-1]]
                        new_weights_grad2_2 = torch.zeros([w2.shape[0], w2.shape[1] + self.new_ns[i-1]])[:, 0:self.new_ns[i-1]]
                        w2 = Variable(torch.cat((w2, new_weights2_2), dim=1))
                        wg2 = Variable(torch.cat((wg2, new_weights_grad2_2), dim=1))

                    if torch.cuda.is_available():
                        w2, wg2, b2, bg2 = w2.cuda(), wg2.cuda(), b2.cuda(), bg2.cuda()
                        self.hebb_values[i] = hebbs.cuda()
                        self.hebb_values_neurites[i] = hebbs_neurites.cuda()

                    self.fcs[i].weight = nn.Parameter(Variable(w2).cuda())
                    self.fcs[i].weight.grad = nn.Parameter(Variable(wg2).cuda())
                    self.fcs[i].bias = nn.Parameter(Variable(b2).cuda())
                    self.fcs[i].bias.grad = nn.Parameter(Variable(bg2).cuda())
                    self.fcs[i].in_features = wg2.shape[1]
                    self.fcs[i].out_features = wg2.shape[0]

            w3 = self.fcs[-1].weight.data.copy_(self.fcs[-1].weight.data).cpu()
            wg3 = self.fcs[-1].weight.grad.data.copy_(self.fcs[-1].weight.grad.data).cpu()
            new_weights3 = torch.zeros([w3.shape[0], w3.shape[1] + self.new_ns[-1]])
            new_weights3 = self.init_function(new_weights3)[:, 0:self.new_ns[-1]]
            new_weights_grad3 = torch.zeros([w3.shape[0], w3.shape[1] + self.new_ns[-1]])[:, 0:self.new_ns[-1]]
            w3 = Variable(torch.cat((w3, new_weights3), dim=1))
            wg3 = Variable(torch.cat((wg3, new_weights_grad3), dim=1))
            if torch.cuda.is_available():
                w3 = w3.cuda()
                wg3 = wg3.cuda()
            self.fcs[-1].weight = nn.Parameter(w3)
            self.fcs[-1].weight.grad = nn.Parameter(wg3)
            self.fcs[-1].in_features = len(self.fcs[-1].bias)

        else:
            print("ERROR")

    def batch_norm_layers(self, x, i):
        if self.bn:
            self.bns[i] = nn.BatchNorm1d(self.fcs[i].out_features)
            if torch.cuda.is_available():
                self.bns[i] = self.bns[i].cuda()
            x = self.bns[i](x)
        return x

    def forward(self, x, trainable=True):
        self.last_epoch += 1
        xs = []
        if type(x) == torch.Tensor and not torch.cuda.is_available():
            x = Variable(torch.FloatTensor(x.data.numpy()), requires_grad=True)
        if len(list(x.shape)) > 2:
            x = x.view(x.shape[0], -1)
        valid_bool = torch.Tensor(self.valid_bool)
        if torch.cuda.is_available():
            valid_bool = valid_bool.cuda()
        x = torch.mul(x, valid_bool)
        x_input = Variable(x.data.copy_(x.data))
        if torch.cuda.is_available():
            x_input = x_input.cuda()
        xs += [x_input.data.copy_(x_input.data)]

        if self.is_conv:
            x = self.forward_conv(trainable=trainable)
        for i in range(len(self.gt)):
            #muls = [torch.mul(self.fcs[i].weight.data, x[j, :]) for j in range(x.shape[0])]
            #mul2 = Variable(torch.sum(torch.stack(muls), dim=0), requires_grad=True)
            #if torch.cuda.is_available():
            #    mul2 = mul2.cuda()
            print(self.fcs[i].weight.data)
            x = self.fcs[i](x)
            print(i, x[0][0])
            x = self.batch_norm_layers(x, i)
            x = F.relu(x)
            xs += [x.data.copy_(x.data)]

            #self.add_hebb_neurites(mul2, i)
            print(i, x[0][0])
            #self.add_hebb_neurons(x, i)
            x = F.dropout(x, training=trainable)
        x = self.fcs[-1](x)
        xs += [x.data.copy_(x.data)]
        #self.add_hebb_neurons_input(xs)

        return x

    def forward_conv(self, trainable=True):
        x = None
        for i in range(len(self.convs)):
            if self.wn:
                x = nn.utils.weight_norm(x)
            x = self.convs[i](x)
            if self.pooling_layers[i] == 1:
                if self.relu:
                    x = self.pool(x)
                    x = F.relu(x)
            elif self.pooling_layers[i] == 0:
                if self.relu:
                    x = F.relu(x)
            if self.bn:
                self.convs_bn[i] = nn.BatchNorm2d(self.convs[i].out_channels)
                x = self.convs_bn[i](x)
            x = F.dropout2d(x, training=trainable)
            hebbs_conv = self.hebb_values_conv[i].data.copy_(self.hebb_values_conv[i].data)
            hebbs_conv = Variable(hebbs_conv)
            val_to_add = torch.sum(hebb_values_transform_conv(x, self.hebb_rates_conv[i]), dim=0) / (
                    x.shape[1] * x.shape[2] * x.shape[3])
            if torch.cuda.is_available():
                val_to_add = val_to_add.cuda()
                hebbs_conv = hebbs_conv.cuda()
            self.hebb_values_conv[i] = torch.add(hebbs_conv, val_to_add)

        return x.squeeze()

    def add_conv_units(self, new_conv_channels, keep_grad=True, init="he", clip_max=100000):
        # TODO augment by a factor, e.g. x2. like that the archtecture would be kept
        hebbs = self.hebb_values_conv
        hebb_zeros = Variable(torch.zeros(new_conv_channels))

        for i in range(len(new_conv_channels)):
            if new_conv_channels[i] > 0:
                hebbs[i] = torch.cat((hebbs[i],))
        w1 = None
        b1 = None
        wg1 = None
        bg1 = None
        w2s = [[] for _ in len(hebb_zeros)]
        b2s = [[] for _ in len(hebb_zeros)]
        wg2s = [[] for _ in len(hebb_zeros)]
        bg2s = [[] for _ in len(hebb_zeros)]
        wg3 = None
        w3 = None

        if keep_grad and init == "he":
            b1 = self.convs[0].bias.data
            w1 = self.convs[0].weight.data
            if new_conv_channels[0] > 0 and len(b1) <= clip_max:
                print("New neurons with kaiming init")
                w_zeros1 = torch.zeros([w1.shape[0] + new_conv_channels[0], w1.shape[1], w1.shape[2], w1.shape[3]])
                wg_zeros1 = torch.zeros([wg1.shape[0] + new_conv_channels[0], wg1.shape[1], wg1.shape[2], wg1.shape[3]])
                new_weights1 = self.init_function(w_zeros1)[0:new_conv_channels[0]]
                new_biases1 = torch.zeros(new_conv_channels[0])

                w1 = torch.cat((w1, new_weights1), dim=0)
                b1 = torch.cat((b1.data, new_biases1))
                wg1 = wg1
                wg1 = torch.cat((wg1, self.init_function(wg_zeros1)[0:new_conv_channels[0]]), dim=0)
    
                b1.grad.data = torch.cat((b1.grad.data, torch.zeros(new_conv_channels[0])))
    
                self.convs[0].out_channels = len(b1)
                self.planes[1] = len(b1.data)
    
            for i in range(1, len(new_conv_channels)):
                b2s[i] = self.convs[i].bias.data
                bg2s[i] = self.convs[i].bias.grad.data
                w2s[i] = self.convs[i].weight.data
                wg2s[i] = self.convs[i].weight.grad.data

                if new_conv_channels[i] > 0 and len(b2s[i]) < clip_max:
                    print("New neurons with kaiming init")
                    w_zeros2_1 = torch.zeros([w2s[i].shape[0], w2s[i].shape[1] + new_conv_channels[i - 1],
                                              wg2s[i].shape[2], wg2s[i].shape[3]])
                    w_zeros2_2 = torch.zeros([w2s[i].shape[0] + new_conv_channels[i], w2s[i].shape[1],
                                              w2s[i].shape[2], w2s[i].shape[3]])
                    wg_zeros2_1 = torch.zeros([wg2s[i].shape[0], new_conv_channels[i - 1],
                                              wg2s[i].shape[2], wg2s[i].shape[3]])
                    wg_zeros2_2 = torch.zeros([new_conv_channels[i], wg2s[i].shape[1],
                                               wg2s[i].shape[2], wg2s[i].shape[3]])
                    b_zeros2 = torch.zeros(new_conv_channels[i])
                    b2s[i] = torch.cat((b2s[i], b_zeros2))
                    w2s[i] = torch.cat((w2s[i], self.init_function(w_zeros2_1)[:, 0:new_conv_channels[i - 1]]), dim=1)
                    w2s[i] = torch.cat((w2s[i], self.init_function(w_zeros2_2)[0:new_conv_channels[i], :]), dim=0)
    
                    bg2s[i] = torch.cat((bg2s[i], b_zeros2))
                    wg2s[i] = torch.cat((wg2s[i], wg_zeros2_1), dim=1)
                    wg2s[i] = torch.cat((wg2s[i], wg_zeros2_2), dim=0)
                    self.planes[i + 1] = len(bg2s[i])
    
                else:
                    print("Already the max neurons. Put them on another layer or place new layer")
        else:
            print("ERROR")

    def replace_neurons(self):
        pass

    def pruning_conv(self, gt_convs, min_neurons=4):
        hebb_conv = self.hebb_values_conv[0].data.copy_(self.hebb_values_conv[0].data)
        to_keep = hebb_conv > float(gt_convs[0])
        to_keep_array = to_keep == 1
        indices_neurons1 = indices_h_conv(to_keep_array)
        if len(indices_neurons1) < min_neurons:
            # TODO Replace neurons that could not be removed?
            print("Minimum neurons on layer 1")
            indices_neurons1 = indices_h_conv(torch.sort(hebb_conv)[1] < min_neurons)
        self.hebb_values_conv[0] = Variable(hebb_conv[indices_neurons1])

        w1 = self.convs[0].weight
        b1 = self.convs[0].bias
        weight1 = w1.data[indices_neurons1, :]
        bias1 = b1.data[indices_neurons1]
        gw1 = self.convs[0].weight.grad[indices_neurons1, :]
        gb1 = self.convs[0].bias.grad[indices_neurons1]

        self.convs[0].weight = torch.nn.Parameter(weight1)
        self.convs[0].bias = torch.nn.Parameter(bias1)
        self.convs[0].in_channels = len(weight1[0])
        self.convs[0].out_channels = len(weight1)
        self.convs[0].weight.grad = gw1
        self.convs[0].bias.grad = gb1

        self.bns[0] = nn.BatchNorm1d(len(self.convs[0].bias))

        for i in range(1, len(gt_convs)):
            hebb2 = self.hebb_values_conv[i].data.copy_(self.hebb_values_conv[i].data)
            to_keep2 = hebb2 > float(gt_convs[i])
            to_keep2_array = to_keep2 == 1
            indices_neurons2 = indices_h_conv(to_keep2_array)
            if len(indices_neurons2) < min_neurons:
                # TODO Replace neurons that could not be removed?
                indices_neurons2 = indices_h_conv(torch.sort(hebb2)[1] < min_neurons)
                print("Minimum neurons on layer ", (i + 1))

            self.hebb_values_conv[i] = Variable(hebb2[indices_neurons2])
            w2 = self.convs[i].weight.data.copy_(self.convs[i].weight.data).cpu().numpy()
            b2 = self.convs[i].bias.data.copy_(self.convs[i].bias.data).cpu().numpy()

            gw2 = self.convs[i].weight.grad.data.copy_(self.convs[i].weight.grad.data).cpu().numpy()
            gb2 = self.convs[i].bias.data.copy_(self.convs[i].bias.grad.data).cpu().numpy()
            gb2 = gb2[indices_neurons2]

            gw2 = gw2[indices_neurons2, :]
            gw2 = gw2[:, indices_neurons1]
            gw2 = torch.from_numpy(gw2)
            gb2 = torch.from_numpy(gb2)

            w2 = w2[indices_neurons2, :]
            w2 = w2[:, indices_neurons1]
            b2 = b2[indices_neurons2]
            w2 = torch.from_numpy(w2)
            b2 = torch.from_numpy(b2)

            if torch.cuda.is_available():
                gw2 = gw2.cuda()
                w2 = w2.cuda()
                gb2 = gb2.cuda()
                b2 = b2.cuda()

            self.convs[i].weight = torch.nn.Parameter(w2)
            self.convs[i].bias = torch.nn.Parameter(b2)
            self.convs[i].in_channels = len(w2[0])
            self.convs[i].out_channels = len(w2)
            self.convs[i].weight.grad = torch.nn.Parameter(gw2)
            self.convs[i].bias.grad = torch.nn.Parameter(gb2)
            self.bns[i] = nn.BatchNorm1d(len(self.convs[i].bias))
            indices_neurons1 = indices_neurons2
        fc1_w = self.fcs[i].weight.data.copy_(self.fcs[i].weight.data).cpu().numpy()
        fc1_wg = self.fcs[i].weight.grad.data.copy_(self.fcs[i].weight.grad.data).cpu().numpy()
        fc1_w = fc1_w[:, indices_neurons1]
        fc1_wg = fc1_wg[:, indices_neurons1]
        fc1_w = torch.from_numpy(fc1_w)
        fc1_wg = torch.from_numpy(fc1_wg)
        self.fcs[i].weight = torch.nn.Parameter(fc1_w)
        self.fcs[i].weight.grad = torch.nn.Parameter(fc1_wg)

    def sort_pruning_values(self, n_remove):
        gts = [[]] * len(n_remove)
        for i in range(len(gts)):
            hebb = Variable(self.hebb_values[i].data.copy_(self.hebb_values[i].data))
            sorted_hebb = np.sort(hebb.data)
            gts[i] = sorted_hebb[n_remove[i]]
        return gts

"""
import math
inf = math.inf
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1, 1)
a, b = 0, inf
mean, var, skew, kurt = truncnorm.stats(a, b, moments='mvsk')
x = np.linspace(truncnorm.ppf(0.01, a, b),
truncnorm.ppf(0.99, a, b), 100)
ax.plot(x, truncnorm.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='truncnorm pdf')

rv = truncnorm(a, b)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

vals = truncnorm.ppf([0.001, 0.5, 0.999], a, b)
np.allclose([0.001, 0.5, 0.999], truncnorm.cdf(vals, a, b))

r = truncnorm.rvs(a, b, size=1000)

ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()




random_normal = np.random.normal(0, 1, 100000000)
np.mean(random_normal)

"""