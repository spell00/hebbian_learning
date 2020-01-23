import torch.nn as nn
import torch.nn.functional as F
from models.NeuralNet import NeuralNet
import torch
from models.discriminative.artificial_neural_networks.hebbian_network.hebbLayers import HebbLayersMLP
import pandas as pd
from torch.autograd import Variable
from dimension_reduction.ordination import ordination2d
from utils.utils import create_missing_folders
import matplotlib.pyplot as plt
import pylab
from torch.nn import init
import numpy as np
from utils.plots import histograms_hidden_layers

def plot_performance(loss_total, accuracy, labels, results_path, filename="NoName", verbose=0, std_loss=None, std_accuracy=None):
    """

    :param loss_total:
    :param loss_labelled:
    :param loss_unlabelled:
    :param accuracy:
    :param labels:
    :param results_path:
    :param filename:
    :param verbose:
    :return:
    """
    fig2, ax21 = plt.subplots()
    n = list(range(len(accuracy["train"])))
    try:
        ax21.plot(loss_total["train"], 'b-', label='Train total loss:' + str(len(labels["train"])))  # plotting t, a separately
        ax21.plot(loss_total["valid"], 'g-', label='Valid total loss:' + str(len(labels["valid"])))  # plotting t, a separately
        #ax21.plot(values["valid"], 'r-', label='Test:' + str(len(labels["valid"])))  # plotting t, a separately
    except:
        ax21.plot(loss_total["train"], 'b-', label='Train total loss:')  # plotting t, a separately
        ax21.plot(loss_total["valid"], 'g-', label='Valid total loss:')  # plotting t, a separately
    if std_accuracy is not None:
        ax21.errorbar(x=n, y=loss_total["train"], yerr=[np.array(std_loss["train"]), np.array(std_loss["train"])],
                      c="b", label='Train')  # plotting t, a separately
    if std_accuracy is not None:
        ax21.errorbar(x=n, y=loss_total["valid"], yerr=[np.array(std_loss["valid"]), np.array(std_loss["valid"])],
                      c="g", label='Valid')  # plotting t, a separately

    ax21.set_xlabel('epochs')
    ax21.set_ylabel('Loss')
    handles, labels = ax21.get_legend_handles_labels()
    ax21.legend(handles, labels)
    ax22 = ax21.twinx()

    #colors = ["b", "g", "r", "c", "m", "y", "k"]
    # if n_list is not None:
    #    for i, n in enumerate(n_list):
    #        ax22.plot(n_list[i], '--', label="Hidden Layer " + str(i))  # plotting t, a separately
    ax22.set_ylabel('Accuracy')
    ax22.plot(accuracy["train"], 'c--', label='Train')  # plotting t, a separately
    ax22.plot(accuracy["valid"], 'k--', label='Valid')  # plotting t, a separately
    if std_accuracy is not None:
        ax22.errorbar(x=n, y=accuracy["train"], yerr=[np.array(std_accuracy["train"]), np.array(std_accuracy["train"])],
                      c="c", label='Train')  # plotting t, a separately
    if std_accuracy is not None:
        ax22.errorbar(x=n, y=accuracy["valid"], yerr=[np.array(std_accuracy["valid"]), np.array(std_accuracy["valid"])],
                      c="k", label='Valid')  # plotting t, a separately

    handles, labels = ax22.get_legend_handles_labels()
    ax22.legend(handles, labels)

    fig2.tight_layout()
    # pylab.show()
    if verbose > 0:
        print("Performance at ", results_path)
    create_missing_folders(results_path + "/plots/")
    pylab.savefig(results_path + "/plots/" + filename)
    plt.show()
    plt.close()



class MLP(NeuralNet):
    def __init__(self, input_size, input_shape, indices_names, h_dims, num_classes, gt_input=-100, gt=-1000, a_dim=0,
                 is_hebb_layers=False, batch_norm=True, dropout=0.5, is_clamp=True, show_pca_train=False, iw=1, mc=1,
                 extra_class=True, l1=0., l2=0., early_stopping=100, is_pruning=False):
        """

        :param input_size:
        :param input_shape:
        :param indices_names:
        :param h_dims:
        :param num_classes:
        :param gt_input:
        :param gt:
        :param a_dim:
        :param hebb_layers:
        :param batch_norm:
        :param dropout:
        :param is_clamp:
        :param show_pca_train:
        :param iw:
        :param mc:
        """
        super(MLP, self).__init__()
        self.is_pruning = is_pruning
        self.epoch = 0
        self.a_dim = a_dim
        #self.valid_bool = [1 for _ in range(input_size)]
        self.valid_bool = None
        self.iw = iw
        self.early_stopping = early_stopping
        self.l1 = l1
        self.l2 = l2
        self.mc = mc
        self.show_pca_train = show_pca_train
        self.input_shape = input_shape
        self.is_hebb_layers = is_hebb_layers
        self.is_clamp = is_clamp
        self.extra_class = extra_class
        self.input_size = input_size
        print("self.input_size", input_size)
        layers_dims = [self.input_size+a_dim]+h_dims
        self.num_classes = num_classes
        if extra_class:
            self.num_classes += 1
        fc_layers = [nn.Linear(layers_dims[i], layers_dims[i+1])
                     for i in range(len(layers_dims)-1)] + [nn.Linear(h_dims[-1], self.num_classes)]
        bn = [nn.BatchNorm1d(layers_dims[i+1]) for i in range(len(layers_dims)-1)]
        self.fcs = nn.ModuleList(fc_layers).cuda()
        if batch_norm:
            self.bn = nn.ModuleList(bn).cuda()
        else:
            self.bn = None
        self.dropout = dropout
        if self.is_hebb_layers:
            self.hebb_layers = HebbLayersMLP(input_size, input_shape, indices_names, self.num_classes, h_dims, lambd=1.,
                                             is_pruning=is_pruning, a_dim=a_dim,
                                             hebb_rates=[0. for _ in range(len(h_dims))],
                                             gt_input=gt_input,
                                             gt=[gt for _ in range(len(h_dims))],
                                             gt_neurites=[gt for _ in range(len(h_dims))],
                                             hebb_rates_neurites=[0. for _ in range(len(h_dims))],
                                             hebb_rates_multiplier=[0. for _ in range(len(h_dims))],
                                             new_ns=[10. for _ in range(len(h_dims))])
            self.hebb_layers.bn_input = nn.BatchNorm1d(input_size+a_dim)
        self.indices_names = indices_names

        self.train_total_loss_history = []
        self.train_accuracy_history = []
        self.valid_total_loss_history = []
        self.valid_accuracy_history = []
        self.hebb_input_values_history = []

        self.train_total_loss_histories = []
        self.train_accuracy_histories = []
        self.valid_total_loss_histories = []
        self.valid_accuracy_histories = []
        self.hebb_input_values_histories = []

    def glorot_init(self):
        self.epoch = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.train_total_loss_history = []
        self.train_accuracy_history = []
        self.valid_total_loss_history = []
        self.valid_accuracy_history = []
        self.hebb_input_values_history = []
        self.cuda()


    def get_n_params(model):
        """

        :return:
        """
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def set_layers(self):

        return

    def run(self, n_epochs, verbose=1, clip_grad=0, is_input_pruning=False, start_pruning=3, show_progress=20,
            is_balanced_relu=False, plot_progress=20, hist_epoch=20, all0=False, overall_mean=False):

        """

        :param n_epochs:
        :param verbose:
        :param clip_grad:
        :param is_input_pruning:
        :param start_pruning:
        :return:
        """
        self.is_balanced_relu = is_balanced_relu
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, verbose=True, cooldown=100,
                                                               patience=100)

        best_loss = 100000
        early = 0
        best_accuracy = 0

        hparams_string = "/".join(["lr"+str(self.lr)])


        involment_df = pd.DataFrame(index=self.indices_names)
        print("Log file created: ",  "logs/" + self.__class__.__name__ + "_parameters.log")
        file_parameters = open("logs/" + self.__class__.__name__ + "_parameters.log", 'w+')
        #print("file:", file_parameters)
        print(*("n_samples:", len(self.train_loader)), sep="\t", file=file_parameters)
        print("Number of classes:", self.num_classes, sep="\t", file=file_parameters)

        print("Total parameters:", self.get_n_params(), file=file_parameters)
        print("Total:", self.get_n_params(), file=file_parameters)
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape, sep="\t", file=file_parameters)
        file_parameters.close()

        print("Log file created: ",  "logs/" + self.__class__.__name__ + "_involvment.log")
        file_involvment = open("logs/" + self.__class__.__name__ + "_involvment.log", 'w+')
        print("started", file=file_involvment)
        file_involvment.close()
        print("Log file created: ",  "logs/" + self.__class__.__name__ + ".log")
        file = open("logs/" + self.__class__.__name__ + ".log", 'w+')
        file.close()
        print("Labeled shape", len(self.train_loader))
        hebb_round = 1
        for _ in range(self.epoch, n_epochs):
            file = open("logs/" + self.__class__.__name__ + ".log", 'a+')
            file_involvment = open("logs/" + self.__class__.__name__ + "_involvment.log", 'a+')
            self.epoch += 1
            self.train()
            total_loss, accuracy, accuracy_total = (0, 0, 0)

            print("epoch", self.epoch, file=file)
            if verbose > 0:
                print("epoch", self.epoch)
            c = 0
            for i, (x, y) in enumerate(self.train_loader):
                c += len(x)
                # progress = 100 * c / len(self.train_loader) / self.batch_size
                #print("Progress: {:.2f}%".format(progress))

                x, y = Variable(x), Variable(y)

                if torch.cuda.is_available():
                    # They need to be on the same device and be synchronized.
                    x, y = x.cuda(), y.cuda()
                if self.epoch % hist_epoch == 0 and i == 1:
                    is_hist = True
                else:
                    is_hist = False

                logits = self(x, valid_bool=self.valid_bool, input_pruning=is_input_pruning,
                              start_pruning=start_pruning, is_balanced_relu=is_balanced_relu, is_hist=is_hist,
                              all0=all0, overall_mean=overall_mean)
                try:
                    targets = torch.max(y, 1)[1].long()
                except:
                    targets = y

                classication_loss = F.cross_entropy(logits, targets)

                params = torch.cat([x.view(-1) for x in self.parameters()])
                l1_regularization = self.l1 * torch.norm(params, 1)
                l2_regularization = self.l2 * torch.norm(params, 2)
                loss = classication_loss + l1_regularization + l2_regularization

                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem.
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                else:
                    pass
                total_loss += loss.item()

                _, pred_idx = torch.max(logits, 1)
                try:
                    _, lab_idx = torch.max(y, 1)
                    accuracy_total += torch.mean((pred_idx.data[0] == lab_idx.data[0]).float())
                    accuracy += torch.mean((pred_idx.data[0] == lab_idx.data[0]).float())

                except:
                    lab_idx = y
                    accuracy_total += torch.mean((pred_idx.data[0] == lab_idx.data).float())
                    accuracy += torch.mean((pred_idx.data[0] == lab_idx.data).float())

                optimizer.step()
                optimizer.zero_grad()

                del loss, x, y

            print(self.fcs, file=file)

            self.eval()
            m = len(self.train_loader)

            if self.epoch % plot_progress == 0:
                self.train_total_loss_history += [(total_loss / m)]
                self.train_accuracy_history += [(accuracy / m)]

            print("Epoch: {}".format(self.epoch), sep="\t", file=file)
            print("[Train]\t\t Loss: {:.2f}, accuracy: {:.4f}".format(total_loss / m, accuracy_total / m),
                  sep="\t", file=file)
            if verbose > 0:
                print("[Train]\t\t Loss: {:.2f}, accuracy: {:.4f}".format(total_loss / m, accuracy_total / m))

            total_loss, accuracy, accuracy_total = (0, 0, 0)

            for x, y in self.valid_loader:

                c += len(x)
                # progress = c / len(self.train_loader)
                # print("Progress: {:.2f}%".format(progress))

                x, y = Variable(x), Variable(y)

                if torch.cuda.is_available():
                    # They need to be on the same device and be synchronized.
                    x, y = x.cuda(), y.cuda()

                # Add auxiliary classification loss q(y|x)

                logits = self(x, valid_bool=self.valid_bool, input_pruning=is_input_pruning,
                              start_pruning=start_pruning, is_balanced_relu=is_balanced_relu, all0=all0,
                              overall_mean=overall_mean)
                try:
                    targets = torch.max(y, 1)[1].long()
                except:
                    targets = y

                classication_loss = F.cross_entropy(logits, targets)

                params = torch.cat([x.view(-1) for x in self.parameters()])
                l1_regularization = self.l1 * torch.norm(params, 1)
                l2_regularization = self.l2 * torch.norm(params, 2)
                loss = classication_loss + l1_regularization + l2_regularization

                # `clip_grad_norm` helps prevent the exploding gradient problem.
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                else:
                    pass
                total_loss += loss.item()
                _, pred_idx = torch.max(logits, 1)

                try:
                    _, lab_idx = torch.max(y, 1)
                    accuracy_total += torch.mean((pred_idx.data == lab_idx.data).float())
                    accuracy += torch.mean((pred_idx.data == lab_idx.data).float())

                except:
                    lab_idx = y
                    accuracy_total += torch.mean((pred_idx.data == lab_idx.data).float())
                    accuracy += torch.mean((pred_idx.data == lab_idx.data).float())

                optimizer.step()
                optimizer.zero_grad()

                del loss, x, y

            m = len(self.valid_loader)
            print("[Validation]\t J_a: {:.2f}, accuracy: {:.4f}".format(total_loss / m,
                  accuracy / m), sep="\t", file=file)
            if verbose > 0:
                print("[Validation]\t J_a: {:.2f}, accuracy: {:.4f}".format(total_loss / m, accuracy / m))
            #m = len(self.test_loader)

            if self.epoch % plot_progress == 0:
                self.valid_total_loss_history += [(total_loss / m)]
                self.valid_accuracy_history += [(accuracy / m)]
            if self.epoch % hebb_round == 0 and self.epoch != 0:
                if self.is_hebb_layers:
                    self.fcs, self.valid_bool, self.bn = self.hebb_layers.compute_hebb(total_loss, self.epoch,
                                                results_path=self.results_path, fcs=self.fcs, verbose=3)
                    alive_inputs = sum(self.valid_bool)
                    if alive_inputs < len(self.valid_bool):
                        print("Current input size:", alive_inputs, "/", len(self.valid_bool))

                    hebb_input_values = self.hebb_layers.hebb_input_values

                    # The last positions are for the auxiliary network, if using auxiliary deep generative model
                    if self.a_dim > 0:
                        involment_df = pd.concat((involment_df, pd.DataFrame(hebb_input_values.detach().cpu().numpy()
                                                                    [:-self.a_dim], index=self.indices_names)), axis=1)
                    else:
                        involment_df = pd.concat((involment_df, pd.DataFrame(hebb_input_values.detach().cpu().numpy(),
                                                                             index=self.indices_names)), axis=1)
                    involment_df.columns = [str(a) for a in range(involment_df.shape[1])]
                    last_col = str(int(involment_df.shape[1])-1)
                    print("epoch", self.epoch, "last ", last_col, file=file_involvment)
                    print(involment_df.sort_values(by=[last_col], ascending=False), file=file_involvment)


            # early-stopping
            if (accuracy > best_accuracy or total_loss < best_loss):
                #print("BEST LOSS!", total_loss / m)
                early = 0
                best_loss = total_loss
                #self.save_model()

            else:
                early += 1
                if early > self.early_stopping:
                    break

            if self.epoch % plot_progress == 0:
                total_losses_histories = {"train": self.train_total_loss_history, "valid": self.valid_total_loss_history}
                accuracies_histories = {"train": self.train_accuracy_history, "valid": self.valid_accuracy_history}
                labels = {"train": self.labels_train, "valid": self.labels_test}
                if self.epoch % show_progress == 0 and self.epoch % hebb_round == 0 and self.epoch != 0:
                    plot_performance(loss_total=total_losses_histories,
                                 accuracy=accuracies_histories,
                                 labels=labels,
                                 results_path=self.results_path + "/" + hparams_string + "/",
                                 filename=self.dataset_name)
            scheduler.step(total_loss)
            file.close()
            file_involvment.close()

            del total_loss, accuracy
        self.train_total_loss_histories += [self.train_total_loss_history]
        self.train_accuracy_histories += [self.train_accuracy_history]
        self.valid_total_loss_histories += [self.valid_total_loss_history]
        self.valid_accuracy_histories += [self.valid_accuracy_history]
        mean_total_losses_histories = {"train": np.mean(np.array(self.train_total_loss_histories), axis=0), "valid": np.mean(np.array(self.valid_total_loss_histories), axis=0)}
        var_losses_histories = {"train": np.std(np.array(self.train_total_loss_histories), axis=0), "valid": np.std(np.array(self.valid_total_loss_histories), axis=0)}
        mean_accuracies_histories = {"train": np.mean(np.array(self.train_accuracy_histories), axis=0), "valid": np.mean(np.array(self.valid_accuracy_histories), axis=0)}
        var_accuracies_histories = {"train": np.std(np.array(self.train_accuracy_histories), axis=0), "valid": np.std(np.array(self.valid_accuracy_histories), axis=0)}
        labels = {"train": self.labels_train, "valid": self.labels_test}
        plot_performance(loss_total=mean_total_losses_histories,
                         std_loss=var_losses_histories,
                         accuracy=mean_accuracies_histories,
                         std_accuracy=var_accuracies_histories,
                         labels=labels,
                         results_path=self.results_path + "/" + hparams_string + "/",
                         filename=self.dataset_name)

    def forward(self, x, valid_bool, a=torch.Tensor([]).cuda(), input_pruning=False, start_pruning=-1, is_hist=False,
                is_balanced_relu=False, all0=False, overall_mean=False, is_conv=False):
        """
        :param x:
        :param a:
        :param valid_bool:
        :return:
        """
        if valid_bool is not None and self.epoch >= start_pruning and start_pruning > -1:
            if type(valid_bool) == list:
                valid_bool = torch.Tensor(valid_bool).cuda()
            x = x.float() * valid_bool.float()
        x = torch.cat([x.float(), a], dim=1)
        if self.is_hebb_layers:
            xs = []
            xs += [x.data.copy_(x.data)]
        if not is_conv:
            x = x.view(x.shape[0], -1)
        for i, fc in enumerate(self.fcs[:-1]):
            x = fc(x)
            if is_hist:
                # TODO provide histogram for a single neuron
                histograms_hidden_layers(xs=x.data.copy_(x.data).detach().cpu().numpy().flatten(),
                                         results_path=self.results_path, is_mean=False, epoch=self.epoch, depth=i,
                                         activated=False, normalized=False, bins=20, flat=True)
                #histograms_hidden_layers(xs=x.data.copy_(x.data).detach().cpu().numpy(),
                #                         results_path=self.results_path, is_mean=False, epoch=self.epoch, depth=i,
                #                         activated=False, normalized=False, bins=40, flat=False)
                histograms_hidden_layers(xs=x.data.copy_(x.data).detach().cpu().numpy(),
                                         results_path=self.results_path, is_mean=False, epoch=self.epoch, depth=i,
                                         activated=False, normalized=False, bins=10, flat=False, neuron=0)
            if self.bn is not None:
                x = self.bn[i](x)
            if is_hist:
                mu = np.mean(x.data.detach().cpu().numpy().flatten())
                var = np.var(x.data.detach().cpu().numpy().flatten())

                # TODO provide histogram for a single neuron; mean will be 0? Because it is not exactly 0 now...
                # should it be 0?
                histograms_hidden_layers(xs=x.data.copy_(x.data).detach().cpu().numpy().flatten(),
                                         results_path=self.results_path, is_mean=False, epoch=self.epoch, depth=i,
                                         activated=False, normalized=True, bins=20, flat=True)
                #histograms_hidden_layers(xs=x.data.copy_(x.data).detach().cpu().numpy(),
                #        overall_mean                 results_path=self.results_path, is_mean=False, epoch=self.epoch, depth=i,
                #                         activated=False, normalized=True, bins=40, flat=False)
                histograms_hidden_layers(xs=x.data.copy_(x.data).detach().cpu().numpy(),
                                         results_path=self.results_path, is_mean=False, epoch=self.epoch, depth=i,
                                         activated=False, normalized=True, bins=10, flat=False, neuron=0)

            if is_balanced_relu:
                x = balance_relu(x, hyper_balance=1., all0=all0, overall_mean=overall_mean)
            else:
                x = F.relu(x)
            if is_hist:
                histograms_hidden_layers(xs=x.data.copy_(x.data).detach().cpu().numpy().flatten(),
                                         results_path=self.results_path, is_mean=False, epoch=self.epoch, depth=i,
                                         activated=True, normalized=True, mu=mu, var=var, bins=60, flat=True)
                #histograms_hidden_layers(xs=x.data.copy_(x.data).detach().cpu().numpy(),
                #                         results_path=self.results_path, is_mean=False, epoch=self.epoch, depth=i,
                #                         activated=True, normalized=True, mu=mu, var=var, bins=120, flat=False)
                histograms_hidden_layers(xs=x.data.copy_(x.data).detach().cpu().numpy(),
                                         results_path=self.results_path, is_mean=False, epoch=self.epoch, depth=i,
                                         activated=True, normalized=True, mu=mu, var=var, bins=20, flat=False, neuron=0)

            if self.dropout > 0.:
                x = F.dropout(x, self.dropout)
            if self.is_hebb_layers:
                self.hebb_layers.add_hebb_neurons(x, i)
                xs += [x.data.copy_(x.data)]

        x = F.softmax(self.fcs[-1](x), dim=-1)
        #if self.is_hebb_layers:
        #    self.hebb_layers.add_hebb_neurons_input(xs, self.fcs, clamp=self.is_clamp)
        return x


def balance_relu(x, hyper_balance=1., overall_mean=False, all0=False):
    if overall_mean:
        sum_a = len(x[x > 0.])
        tot = len(x)
        #sum_b = len(x[x == 0.])
        ratio = tot / (sum_a+1e-8)
        mu_x = torch.mean(x)
        x[x == 0.] = -mu_x*ratio
        #x = torch.Tensor([x_i if x_i > 0. else -mu_x*ratio for x_i in tmp]).cuda().reshape(x.shape)
    else:
        mu_x = torch.mean(x, 1)
        for i, x_i in enumerate(x):
            sum_a = len(x_i[x_i == 0.])
            sum_b = len(x_i[x_i > 0.])
            tot = len(x_i)
            #if true, the total of all neurons will be true. Otherwise, each neuron's sum of a batch should be 0
            if all0:
                ratio = tot / (sum_a+1e-8)
            else:
                ratio = sum_a / (sum_b+1e-8)

            x_i[x_i == 0.] = -mu_x[i]*ratio*hyper_balance
            x[i] = x_i

    return x

