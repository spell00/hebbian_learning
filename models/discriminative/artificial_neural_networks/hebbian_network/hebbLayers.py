from torch.autograd import Variable
import torch
import torch.nn as nn
from models.NeuralNet import NeuralNet
from models.discriminative.artificial_neural_networks.hebbian_network.utils import hebb_values_transform, hebb_array_transform
import numpy as np
import matplotlib.pyplot as plt
from models.discriminative.artificial_neural_networks.hebbian_network.utils import indices_h, indices_h_conv
from utils.utils import create_missing_folders


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


class HebbLayersMLP(NeuralNet):
    def __init__(self, input_size, input_shape, indices_names, num_classes, Ns, hebb_rates, gt, hebb_rates_neurites, hebb_rates_multiplier,
                 new_ns, kernels=None, gt_neurites=None, lambd=1., clamp_max=1000000, clamp_min=-1000000, gt_input=-1000,
                 padding_pooling=1, padding_no_pooling=1, hebb_max_value=10000, a_dim=0,
                 how_much_more=1.0, hyper_count=100, keep_grad=True, is_pruning=True,
                 hb=True, is_conv=False, schedule_value=0, gt_convs=None, new_ns_convs=None):

        super().__init__()
        self.init_function = torch.nn.init.kaiming_normal_
        self.input_shape = input_shape
        self.input_size = input_size
        try:
            self.n_channels = input_shape[0]
        except:
            self.n_channels = input_shape

        self.hebb_log = open("logs/" + self.__class__.__name__ + "involvment.log", 'w+')
        self.hebb_rate_input = 0
        self.previous_loss = 10000000
        self.best_loss = 10000000
        self.count = 0
        self.indices_names = indices_names
        self.count_down = 0
        self.is_conv = is_conv
        self.gt_input = gt_input
        self.lambd = lambd
        self.hb = hb
        self.hebb_input_values_history = []
        self.keep_grad = keep_grad
        self.is_pruning = is_pruning
        self.a_dim = a_dim
        self.hebb_input_values = Variable(torch.zeros(self.input_size + a_dim))
        if torch.cuda.is_available():
            self.hebb_input_values.cuda()
        self.valid_bool = list(range(self.input_size))
        self.valid_bool_tensor = torch.Tensor(self.valid_bool).cuda()
        self.alive_inputs = list(range(self.input_size))
        self.previous_valid_len = self.input_size
        self.num_classes = num_classes

        # Integers
        self.clamp_max = clamp_max
        self.clamp_min = clamp_min
        self.hyper_count = hyper_count
        self.schedule_value = schedule_value

        # Floats
        self.how_much_more = how_much_more

        self.hebb_max_value = hebb_max_value
        if Ns is not None:
            self.Ns = Ns
        self.n_neurites = [[] for _ in self.Ns]
        # self.n_neurites[0] += [self.input_size * self.Ns[0]]
        # for i in range(len(self.Ns)-1):
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


        # Booleans
        self.descending = True
        self.is_conv = is_conv

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

        list1 = [self.input_size] + self.Ns
        self.list1 = list1
        self.hebb_values_neurites = [torch.zeros(list1[i + 1], list1[i]) for i in range(len(self.Ns))]
        self.original_num_neurites = [int(x.shape[0] * int(x.shape[1])) for x in self.hebb_values_neurites]

        self.hebb_values = [Variable(torch.Tensor([0] * n)) for n in self.Ns]
        self.n_neurons = [[] for _ in self.Ns]

        lenlen = {
            "gt": len(self.gt),
            "hebb_rates_multiplier": len(self.hebb_rates_multiplier),
            "hebb_rates_neurites": len(self.hebb_rates_neurites),
            "hebb_rates": len(self.hebb_rates),
            "new_ns": len(self.new_ns),
            "gt_neurites": len(self.gt_neurites)
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
        # Dicts
        self.labels_dict = {"train": [], "valid": [], "valid": []}
        self.accuracies_dict = {"train": [], "valid": [], "valid": []}
        self.losses_dict = {"train": [], "valid": [], "valid": []}

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

    def calculate_neurons_usage(self):
        pass

    def calculate_neurites_usage(self):
        pass

    def print_parameters(self, fcs):
        print("Optimizer", self.optimizer)
        print("fcs", fcs)
        for i, fc in enumerate(fcs):
            print("fcs", i, ":", fc)
            print("fcs grad", i, ":", fc.weight.grad.shape)
            print("fcs weight", i, ":", fc.weight.shape)
        print("bns[i]", self.bns[i])
        for i, bn in enumerate(self.bns):
            print("bns", i, ":", bn)

    def compute_hebb(self, running_loss, epoch, verbose, fcs, results_path, count_down_limit=10, display_rate=10):
        self.epoch = epoch
        valid_bool = [1. for i in range(self.input_size)]
        print("input_size:", self.input_size)
        print("valid_bool:", len(valid_bool))
        if verbose > 1:
            print("Hebb rates inputs:", self.hebb_rate_input, file=self.hebb_log)
            print("Hebb rates neurons:", self.hebb_rates, file=self.hebb_log)
            print("Hebb rates neurites:", self.hebb_rates_neurites, file=self.hebb_log)
            print("Input layer mean hebb", torch.mean(self.hebb_input_values), file=self.hebb_log)
            print("Input layer min hebb", torch.min(self.hebb_input_values), file=self.hebb_log)
            print("Input layer max hebb", torch.max(self.hebb_input_values), file=self.hebb_log)
            #print("Input layer mean hebb", torch.mean(self.hebb_input_values))
            #print("Input layer min hebb", torch.min(self.hebb_input_values))
            #print("Input layer max hebb", torch.max(self.hebb_input_values))
            print("First layer mean hebb", torch.mean(self.hebb_values[0]), file=self.hebb_log)
            print("First layer min hebb", torch.min(self.hebb_values[0]), file=self.hebb_log)
            print("First layer max hebb", torch.max(self.hebb_values[0]), file=self.hebb_log)
        print("First layer mean hebb", torch.mean(self.hebb_values[0]))
        print("First layer min hebb", torch.min(self.hebb_values[0]))
        print("First layer max hebb", torch.max(self.hebb_values[0]))
        if epoch == 0:
            pass
        elif running_loss < self.previous_loss:
            if self.descending:
                self.count += 1
                if verbose > 2:
                    print("Changing direction: doing better", sep="\t", file=self.hebb_log)
            self.descending = False
            self.count_down = 0

        else:
            if not self.descending:
                if verbose > 2:
                    print("Changing direction: doing worst", sep="\t", file=self.hebb_log)
            else:
                self.count_down += 1
            self.descending = True

        if running_loss < self.best_loss:
            if verbose > 0:
                print("Better loss!", sep="\t", file=self.hebb_log)
            if running_loss < (self.best_loss + self.how_much_more):
                if verbose > 2:
                    print("Count reset to 0", sep="\t", file=self.hebb_log)
                print("Better loss! Count reset to 0.")
                self.count = 0
                self.best_loss = running_loss
            else:
                if verbose > 0:
                    print("Improvement not big enough. Count still going up", sep="\t", file=self.hebb_log)
        print("HYPER COUNT", self.hyper_count, sep="\t", file=self.hebb_log)
        if self.count == self.hyper_count or self.count_down == count_down_limit:
            if verbose > 2:
                print("new ne   urons", sep="\t", file=self.hebb_log)
                if self.count == self.hyper_count:
                    print("Reason: Hyper count reached", sep="\t", file=self.hebb_log)
                else:
                    print("Reason: Worsening limit reached", sep="\t", file=self.hebb_log)
            if self.is_conv:
                self.add_conv_units(new_conv_channels=self.new_ns_convs, keep_grad=True, init="he")
            print("new neurons")
            fcs = self.add_neurons(fcs)
            self.count = 0
        elif self.is_pruning:
            # print("input pruning...")
            #valid_bool, _ = self.input_pruning(results_path)
            if self.is_conv:
                exit("NOT IMPLMENTED")
                # self.pruning_conv()
            fcs, self.bn = self.pruning(fcs)
        if verbose > 0:
            print("count: ", self.count, sep="\t", file=self.hebb_log)
            print("count down: ", self.count_down, sep="\t", file=self.hebb_log)

        self.running_losses.append(running_loss)
        if (epoch > 0):
            if epoch % display_rate == 0 and verbose > 1:
                print("previous accuracy: ", self.previous_loss, sep="\t", file=self.hebb_log)
                print("running_loss: ", running_loss, sep="\t", file=self.hebb_log)
            self.previous_loss = running_loss
            self.previous_acc = self.accuracies_dict["train"]
        return fcs, valid_bool, nn.ModuleList(self.bn).cuda()

    def add_hebb_neurites(self, mul, layer):
        hvals_neurites1 = self.hebb_values_neurites[layer]
        hvals_neurites1 = Variable(hvals_neurites1, requires_grad=False)

        hrate_neurites1 = -torch.mean(mul, 1)

        self.hebb_rates_neurites[layer] = hrate_neurites1
        if torch.cuda.is_available():
            hvals_neurites1 = hvals_neurites1.cuda()
        matrix_to_add = Variable(hebb_values_transform(mul, hrate_neurites1), requires_grad=False)
        if torch.cuda.is_available():
            matrix_to_add = matrix_to_add.cuda()
            hvals_neurites1 = hvals_neurites1.cuda()

        self.hebb_values_neurites[layer] = torch.add(hvals_neurites1, matrix_to_add)
        self.hebb_values_neurites[layer] = torch.clamp(self.hebb_values_neurites[layer], max=self.clamp_max)

    def add_hebb_neurons_input(self, xs, fcs, clamp=False):
        x_input = self.bn_input(xs[0]).cuda()
        x_input[x_input != x_input] = 0
        matmul = xs[-1]
        for i in range(len(fcs)-1).__reversed__():
            matmul = torch.matmul(matmul, fcs[i].weight)
        mul = torch.mul(x_input, matmul)

        mul[mul != mul] = 0

        self.hebb_rate_input = -torch.mean(mul, 1)
        #for j in range(len(self.hebb_rate_input)):
        val_to_add_input = torch.sum(hebb_array_transform(mul, self.hebb_rate_input), dim=0).cuda()
        self.hebb_input_values = torch.add(val_to_add_input.cuda(), self.hebb_input_values.cuda())
        if clamp:
            self.hebb_input_values = torch.clamp(self.hebb_input_values, min=self.clamp_min, max=self.clamp_max)

    def add_hebb_neurons(self, x, i):
        hvals = self.hebb_values[i]
        x[x != x] = 0
        hrate = -torch.mean(x)

        #
        hrate *= self.lambd

        self.hebb_rates[i] = hrate * self.lambd
        vals = Variable(hebb_values_transform(x, hrate), requires_grad=False)
        val_to_add = torch.sum(vals, dim=0)
        if torch.cuda.is_available():
            val_to_add = val_to_add.cuda()
            hvals = hvals.cuda()
        self.hebb_values[i] = torch.add(hvals, val_to_add)
        self.hebb_values[i] = torch.clamp(self.hebb_values[i], max=self.clamp_max)

    def pruning(self, fcs, minimum_neurons=2):
        bn = []
        for i in range(len(self.gt)):
            alive_neurons_out = self.hebb_values[i] > float(self.gt[i])
            indices_alive_neurons_out = indices_h(alive_neurons_out)
            if len(indices_alive_neurons_out) < minimum_neurons:
                indices_alive_neurons_out = indices_h(torch.sort(self.hebb_values[i])[1] < minimum_neurons)
                print("Minimum neurons on layer ", (i + 1), sep="\t", file=self.hebb_log)
                print("Minimum neurons on layer ", (i + 1), sep="\t")
            #self.hebb_values_neurites[i] = self.hebb_values_neurites[i][indices_alive_neurons_out, :]

            w2 = fcs[i].weight.data.copy_(fcs[i].weight.data).cpu().numpy()
            b2 = fcs[i].bias.data.copy_(fcs[i].bias.data).cpu().numpy()
            wg2 = fcs[i].weight.grad.data.copy_(fcs[i].weight.grad.data).cpu().numpy()
            bg2 = fcs[i].bias.grad.data.copy_(fcs[i].bias.grad.data).cpu().numpy()

            bg2 = bg2[indices_alive_neurons_out]
            b2 = b2[indices_alive_neurons_out]

            wg2 = wg2[indices_alive_neurons_out, :]
            w2 = w2[indices_alive_neurons_out, :]

            if i > 0:
                alive_neurons_in = torch.Tensor([True if x > float(self.gt[i - 1]) else False for x in self.hebb_values[i - 1]])
                indices_alive_neurons_in = indices_h(alive_neurons_in)

                #self.hebb_values_neurites[i] = self.hebb_values_neurites[i][:, indices_alive_neurons_in]
                wg2 = wg2[:, indices_alive_neurons_in]
                w2 = w2[:, indices_alive_neurons_in]
                fcs[i].in_features = wg2.shape[1]

            self.Ns[i] = len(b2)
            fcs[i].out_features = len(b2)

            b2 = torch.from_numpy(b2)
            bg2 = torch.from_numpy(bg2)
            w2 = torch.from_numpy(w2)
            wg2 = torch.from_numpy(wg2)

            if torch.cuda.is_available():
                w2 = Variable(w2).cuda()
                wg2 = Variable(wg2).cuda()
                b2 = Variable(b2).cuda()
                bg2 = Variable(bg2).cuda()

            fcs[i].weight = nn.Parameter(w2)
            fcs[i].weight.grad = nn.Parameter(wg2)
            fcs[i].bias = nn.Parameter(b2)
            fcs[i].bias.grad = nn.Parameter(bg2)

            # alive_neurites = self.hebb_values_neurites[i] > self.gt_neurites[i]
            # alive_neurites = torch.Tensor(alive_neurites.data.cpu().numpy()).cuda()

            # fcs[i].weight.data = fcs[i].weight.data * alive_neurites
            # self.n_neurites[i] += [int(torch.sum(alive_neurites))]
            self.hebb_values[i] = self.hebb_values[i][indices_alive_neurons_out]
            bn += [nn.BatchNorm1d(len(self.hebb_values[i]))]

        w3 = fcs[-1].weight.data.copy_(fcs[-1].weight.data).cpu().numpy()
        wg3 = fcs[-1].weight.grad.data.copy_(fcs[-1].weight.grad.data).cpu().numpy()


        try:
            wg3 = wg3[:, indices_alive_neurons_out]
            fcs[-1].in_features = len(indices_alive_neurons_out)
            fcs[-1].weight = nn.Parameter(Variable(torch.from_numpy(w3[:, indices_alive_neurons_out])).cuda())
            fcs[-1].weight.grad = nn.Parameter(Variable(torch.from_numpy(wg3)).cuda())

        except:
            fcs[-1].weight = nn.Parameter(Variable(torch.from_numpy(w3)).cuda())
            fcs[-1].weight.grad = nn.Parameter(Variable(torch.from_numpy(wg3)).cuda())

        if torch.cuda.is_available():
            fcs = fcs.cuda()
        print("Neurons in layers:", self.Ns)
        return fcs, bn

    def input_pruning(self, results_path, min_n_input_dims=20, minimum_neurons=20):
        """
        :param net:
        :param gt:
        :param min_n_input_dims:
        :param minimum_neurons:
        :return:
        """
        hebb_input = self.hebb_input_values.data.copy_(self.hebb_input_values.data).cpu().numpy()
        if len(hebb_input) >= min_n_input_dims:
            to_keep = hebb_input > float(self.gt_input)
            print("min_hebb_value:", self.gt_input)
            valid_indices = indices_h(to_keep)
            if len(valid_indices) < minimum_neurons:
                # TODO Replace neurons that could not be removed?
                valid_indices = indices_h(torch.sort(hebb_input)[1] < minimum_neurons)
                print("Minimum neurons on layer 1", sep="\t", file=self.hebb_log)

            print("previous_valid_len", self.previous_valid_len)
            self.valid_bool = [1. if x in valid_indices else 0. for x in range(self.input_size)]
            self.alive_inputs = [x for x in range(len(hebb_input)) if x in valid_indices]
            alive_inputs = np.array(self.alive_inputs)
            #if len(self.alive_inputs) < self.previous_valid_len:
            masks_path = results_path + "/images/masks/" + str(self.dataset_name) + "/"
            create_missing_folders(masks_path)
            img_path = "_".join(["alive_inputs", str(len(valid_indices)), str(self.epoch), ".png"])
            print("self.n_channels", self.n_channels)
            if len(self.input_shape) == 3:
                print("SAVING MASK at", results_path)
                mask = np.reshape(self.valid_bool, newshape=(28, 28))  # TODO change hard coding
                plt.imsave(masks_path + img_path, mask)
            self.previous_valid_len = len(valid_indices)
            self.valid_bool_tensor = self.valid_bool_tensor * torch.Tensor(self.valid_bool).cuda()
            return self.valid_bool, self.alive_inputs

    def add_neurons(self, fcs):
        for i in range(len(self.new_ns)):
            if self.new_ns[i] > 0:
                self.bn[i] = nn.BatchNorm1d(len(self.bn[i].weight) + int(self.new_ns[i]))
                hebbs = Variable(self.hebb_values[i].data.copy_(self.hebb_values[i].data)).cpu()
                new_neurons = Variable(torch.zeros(int(self.new_ns[i])))
                hebbs = Variable(torch.cat((hebbs, new_neurons)))
                self.Ns[i] = len(hebbs)
                hebbs_neurites = Variable(
                    self.hebb_values_neurites[i].data.copy_(self.hebb_values_neurites[i].data)).cpu()
                new_neurites1 = Variable(torch.zeros(int(self.new_ns[i]), hebbs_neurites.shape[1]))
                hebbs_neurites = Variable(torch.cat((hebbs_neurites, new_neurites1), dim=0))

                w2 = fcs[i].weight.data.copy_(fcs[i].weight.data).cpu()
                b2 = fcs[i].bias.data.copy_(fcs[i].bias.data).cpu()
                wg2 = fcs[i].weight.grad.data.copy_(fcs[i].weight.grad.data).cpu()
                bg2 = fcs[i].bias.grad.data.copy_(fcs[i].bias.grad.data).cpu()
                new_biases2 = torch.zeros(int(self.new_ns[i]))
                b2 = torch.cat((b2, new_biases2))
                bg2 = Variable(torch.cat((bg2, new_biases2)))

                new_weights1 = torch.zeros([w2.shape[0] + int(self.new_ns[i]), w2.shape[1]])
                new_weights1 = torch.nn.init.kaiming_normal_(new_weights1)[0:int(self.new_ns[i]), :]
                new_weights_grad1 = torch.zeros([w2.shape[0] + int(self.new_ns[i]), w2.shape[1]])[0:int(self.new_ns[i]), :]
                w2 = torch.cat((w2, new_weights1), dim=0)
                wg2 = torch.cat((wg2, new_weights_grad1), dim=0)

                if i > 0:
                    new_neurites2 = Variable(torch.zeros(len(hebbs_neurites), int(self.new_ns[i - 1])))
                    hebbs_neurites = Variable(torch.cat((hebbs_neurites, new_neurites2), dim=1))
                    new_weights2_2 = torch.zeros([w2.shape[0], w2.shape[1] + int(self.new_ns[i - 1])])
                    new_weights2_2 = torch.nn.init.kaiming_normal_(new_weights2_2)[:, 0:int(self.new_ns[i - 1])]
                    new_weights_grad2_2 = torch.zeros([w2.shape[0], w2.shape[1] + int(self.new_ns[i - 1])])[:,
                                          0:int(self.new_ns[i - 1])]
                    w2 = Variable(torch.cat((w2, new_weights2_2), dim=1))
                    wg2 = Variable(torch.cat((wg2, new_weights_grad2_2), dim=1))

                if torch.cuda.is_available():
                    w2, wg2, b2, bg2 = w2.cuda(), wg2.cuda(), b2.cuda(), bg2.cuda()
                    self.hebb_values[i] = hebbs.cuda()
                    self.hebb_values_neurites[i] = hebbs_neurites.cuda()

                fcs[i].weight = nn.Parameter(Variable(w2).cuda())
                fcs[i].weight.grad = nn.Parameter(Variable(wg2).cuda())
                fcs[i].bias = nn.Parameter(Variable(b2).cuda())
                fcs[i].bias.grad = nn.Parameter(Variable(bg2).cuda())
                fcs[i].in_features = wg2.shape[1]
                fcs[i].out_features = wg2.shape[0]

        w3 = fcs[-1].weight.data.copy_(fcs[-1].weight.data).cpu()
        wg3 = fcs[-1].weight.grad.data.copy_(fcs[-1].weight.grad.data).cpu()
        new_weights3 = torch.zeros([w3.shape[0], int(w3.shape[1] + int(self.new_ns[i - 1]))])
        new_weights3 = torch.nn.init.kaiming_normal_(new_weights3[:, 0:int(self.new_ns[i - 1])])
        new_weights_grad3 = torch.zeros([w3.shape[0], w3.shape[1] + int(self.new_ns[i - 1])])[:, 0:int(self.new_ns[i - 1])]
        w3 = Variable(torch.cat((w3, new_weights3), dim=1))
        wg3 = Variable(torch.cat((wg3, new_weights_grad3), dim=1))
        if torch.cuda.is_available():
            w3 = w3.cuda()
            wg3 = wg3.cuda()
        fcs[-1].weight = nn.Parameter(w3)
        fcs[-1].weight.grad = nn.Parameter(wg3)
        fcs[-1].in_features = len(fcs[-1].bias)
        return fcs

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
                print("New neurons with kaiming init", sep="\t", file=self.hebb_log)
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
                    print("New neurons with kaiming init", sep="\t", file=self.hebb_log)
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
                    print("Already the max neurons. Put them on another layer or place new layer", sep="\t", file=self.hebb_log)
        else:
            print("ERROR")

    def replace_neurons(self):
        pass

    def pruning_conv(self, fcs, gt_convs, min_neurons=4):
        hebb_conv = self.hebb_values_conv[0].data.copy_(self.hebb_values_conv[0].data)
        to_keep = hebb_conv > float(gt_convs[0])
        to_keep_array = to_keep == 1
        indices_neurons1 = indices_h_conv(to_keep_array)
        if len(indices_neurons1) < min_neurons:
            # TODO Replace neurons that could not be removed?
            print("Minimum neurons on layer 1", sep="\t", file=self.hebb_log)
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
                print("Minimum neurons on layer ", (i + 1), sep="\t", file=self.hebb_log)

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
        fc1_w = fcs[i].weight.data.copy_(fcs[i].weight.data).cpu().numpy()
        fc1_wg = fcs[i].weight.grad.data.copy_(fcs[i].weight.grad.data).cpu().numpy()
        fc1_w = fc1_w[:, indices_neurons1]
        fc1_wg = fc1_wg[:, indices_neurons1]
        fc1_w = torch.from_numpy(fc1_w)
        fc1_wg = torch.from_numpy(fc1_wg)
        fcs[i].weight = torch.nn.Parameter(fc1_w)
        fcs[i].weight.grad = torch.nn.Parameter(fc1_wg)

    def sort_pruning_values(self, n_remove):
        gts = [[]] * len(n_remove)
        for i in range(len(gts)):
            hebb = Variable(self.hebb_values[i].data.copy_(self.hebb_values[i].data))
            sorted_hebb = np.sort(hebb.data)
            gts[i] = sorted_hebb[n_remove[i]]
        return gts

