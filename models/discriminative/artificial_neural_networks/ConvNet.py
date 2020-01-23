import torch.nn as nn
import torch.nn.functional as F
from models.NeuralNet import NeuralNet
import pandas as pd
import torch
from models.generative.autoencoders.vae.GatedConv import GatedConv2d
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from utils.utils import create_missing_folders

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


class ConvNet(NeuralNet):
    def __init__(self, input_shape, h_dims, num_classes, planes, kernels, pooling_layers, indices_names=None, a_dim=0, extra_class=False, is_clamp=False,
                 l1=0, l2=0, early_stopping=100, batch_norm=True):
        """

        :param input_shape:
        :param h_dims:
        :param num_classes:
        :param planes:
        :param kernels:
        :param pooling_layers:
        :param a_dim:
        :param is_clamp:
        :param l1:
        :param l2:
        """
        super(ConvNet, self).__init__()
        layers_dims = [planes[-1] + a_dim]+h_dims
        self.valid_bool = None
        self.input_shape = input_shape
        if indices_names is None:
            indices_names = list(range(np.prod(self.input_shape)))
        self.indices_names = indices_names
        self.extra_class = extra_class
        self.planes = planes
        self.batch_norm = batch_norm
        self.bn = [nn.BatchNorm1d(layer_dim).cuda() for layer_dim in layers_dims[1:]]
        self.l1 = l1
        self.l2 = l2
        self.early_stopping = early_stopping
        self.a_dim = 0
        self.kernels = kernels
        self.pooling_layers = pooling_layers
        self.conv_bn = [nn.BatchNorm2d(planes[i]).cuda() for i in range(1, len(planes))]

        self.convs = nn.ModuleList([GatedConv2d(planes[i], planes[i+1], kernel_size=kernels[i], padding=1, stride=1)
                                    for i in range(len(planes)-1)]).cuda()
        self.denses = nn.ModuleList([nn.Linear(layers_dims[i], layers_dims[i+1]) for i in range(len(layers_dims)-1)]).cuda()
        self.logits = nn.Linear(h_dims[0], num_classes)
        self.pool = nn.MaxPool2d(2, 2)

        self.is_clamp = is_clamp
        self.is_hebb_layers = False

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

    def forward(self, x, a=torch.Tensor([]).cuda(), valid_bool=None, start_pruning=-1):
        if valid_bool is not None and self.epoch >= start_pruning and start_pruning > 0:
            if type(valid_bool) == list:
                valid_bool = torch.Tensor(valid_bool).cuda()
            x = x.float() * valid_bool.float()

        if len(x.shape) == 2:
            x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])

        for i, (conv, conv_bn, kernel, pooling_layer) in enumerate(zip(self.convs, self.conv_bn, self.kernels, self.pooling_layers)):
            x = conv(x)
            conv_bn(x)
            x = F.relu(x)
            if pooling_layer:
                x = self.pool(x)
            x = F.dropout2d(x)
        x = x.squeeze()
        x = torch.cat([x, a], dim=1)
        for i, (dense, bn) in enumerate(zip(self.denses, self.bn)):
            try:
                x = dense(x)
            except:
                exit()
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x)
        x = F.softmax(self.logits(x), dim=-1)
        return x

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, verbose=True, cooldown=100,
                                                               patience=100)

        best_loss = 100000
        early = 0
        best_accuracy = 0

        hparams_string = "/".join(["lr" + str(self.lr)])

        involment_df = pd.DataFrame(index=self.indices_names)
        print("Log file created: ", "logs/" + self.__class__.__name__ + "_parameters.log")
        file_parameters = open("logs/" + self.__class__.__name__ + "_parameters.log", 'w+')
        # print("file:", file_parameters)
        print(*("n_samples:", len(self.train_loader)), sep="\t", file=file_parameters)
        print("Number of classes:", self.num_classes, sep="\t", file=file_parameters)

        print("Total parameters:", self.get_n_params(), file=file_parameters)
        print("Total:", self.get_n_params(), file=file_parameters)
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape, sep="\t", file=file_parameters)
        file_parameters.close()

        print("Log file created: ", "logs/" + self.__class__.__name__ + "_involvment.log")
        file_involvment = open("logs/" + self.__class__.__name__ + "_involvment.log", 'w+')
        print("started", file=file_involvment)
        file_involvment.close()
        print("Log file created: ", "logs/" + self.__class__.__name__ + ".log")
        file = open("logs/" + self.__class__.__name__ + ".log", 'w+')
        file.close()
        print("Labeled shape", len(self.train_loader))
        hebb_round = 1
        for _ in range(self.epoch, n_epochs):
            file = open("logs/" + self.__class__.__name__ + ".log", 'a+')
            file_involvment = open("logs/" + self.__class__.__name__ + "_involvment.log", 'a+')
            self.epoch += 1
            self.train()
            total_loss, accuracy_total = (0, 0)

            print("epoch", self.epoch, file=file)
            if verbose > 0:
                print("epoch", self.epoch)
            c = 0
            for i, (x, y) in enumerate(self.train_loader):
                if verbose > 1:
                    c += len(x)
                    progress = 100 * c / len(self.train_loader) / self.batch_size
                    print("\rProgress: {:.2f}%".format(progress), end="", flush=True)

                x, y = Variable(x), Variable(y)

                if torch.cuda.is_available():
                    # They need to be on the same device and be synchronized.
                    x, y = x.cuda(), y.cuda()
                if self.epoch % hist_epoch == 0 and i == 1:
                    is_hist = True
                else:
                    is_hist = False

                logits = self(x, valid_bool=self.valid_bool)
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
                    accuracy_total += torch.sum((pred_idx.data == lab_idx.data).float())

                except:
                    lab_idx = y
                    accuracy_total += torch.sum((pred_idx.data == lab_idx.data).float())

                optimizer.step()
                optimizer.zero_grad()

                del loss, x, y

            if self.epoch % hebb_round == 0 and self.epoch != 0:
                if self.is_hebb_layers:
                    self.fcs, self.valid_bool = self.hebb_layers.compute_hebb(total_loss, self.epoch,
                                                                              results_path=self.results_path,
                                                                              fcs=self.fcs,
                                                                              verbose=3)
                    alive_inputs = sum(self.valid_bool)
                    if alive_inputs < len(self.valid_bool):
                        print("Current input size:", alive_inputs, "/", len(self.valid_bool))

                    hebb_input_values = self.hebb_layers.hebb_input_values

                    # The last positions are for the auxiliary network, if using auxiliary deep generative model
                    if self.a_dim > 0:
                        involment_df = pd.concat((involment_df, pd.DataFrame(hebb_input_values.detach().cpu().numpy()
                                                                             [:-self.a_dim], index=self.indices_names)),
                                                 axis=1)
                    else:
                        involment_df = pd.concat((involment_df, pd.DataFrame(hebb_input_values.detach().cpu().numpy(),
                                                                             index=self.indices_names)), axis=1)
                    involment_df.columns = [str(a) for a in range(involment_df.shape[1])]
                    last_col = str(int(involment_df.shape[1]) - 1)
                    print("epoch", self.epoch, "last ", last_col, file=file_involvment)
                    print(involment_df.sort_values(by=[last_col], ascending=False), file=file_involvment)

                #print(self.fcs, file=file)

            self.eval()
            m1 = len(self.labels_train)
            if self.epoch % plot_progress == 0:
                self.train_total_loss_history += [(total_loss / len(self.train_loader))]
                self.train_accuracy_history += [(accuracy_total / len(self.labels_train))]

            print("Epoch: {}".format(self.epoch), sep="\t", file=file)
            print("[Train]\t\t Loss: {:.2f}, accuracy: {:.4f}".format(total_loss / len(self.train_loader),
                                                                      accuracy_total / len(self.labels_train)),
                  sep="\t", file=file)
            if verbose > 0:
                print("[Train]\t\t Loss: {:.2f}, accuracy: {:.4f}".format(total_loss / len(self.train_loader),
                                                                          accuracy_total / len(self.labels_train)))

            total_loss, accuracy_total = (0, 0)

            for x, y in self.valid_loader:

                c += len(x)
                # progress = c / len(self.train_loader)
                # print("Progress: {:.2f}%".format(progress))

                x, y = Variable(x), Variable(y)

                if torch.cuda.is_available():
                    # They need to be on the same device and be synchronized.
                    x, y = x.cuda(), y.cuda()

                # Add auxiliary classification loss q(y|x)

                logits = self(x, valid_bool=self.valid_bool)
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
                    accuracy_total += torch.sum((pred_idx.data == lab_idx.data).float())

                except:
                    lab_idx = y
                    accuracy_total += torch.sum((pred_idx.data == lab_idx.data).float())

                optimizer.step()
                optimizer.zero_grad()

                del loss, x, y

            print("[Validation]\t J_a: {:.2f}, accuracy: {:.4f}".format(total_loss / len(self.valid_loader),
                                                                        accuracy_total / len(self.labels_valid)),
                                                                        sep="\t", file=file)
            if verbose > 0:
                print("[Validation]\t J_a: {:.2f}, accuracy: {:.4f}".format(total_loss / len(self.valid_loader),
                                                                            accuracy_total / len(self.labels_valid)))

            m2 = len(self.labels_train)
            if self.epoch % plot_progress == 0:
                self.valid_total_loss_history += [(total_loss / len(self.valid_loader))]
                self.valid_accuracy_history += [(accuracy_total / len(self.labels_train))]

            # early-stopping
            if (accuracy_total > best_accuracy or total_loss < best_loss):
                # print("BEST LOSS!", total_loss / m)
                early = 0
                best_loss = total_loss
                # self.save_model()

            else:
                early += 1
                if early > self.early_stopping:
                    break

            if self.epoch % plot_progress == 0:
                total_losses_histories = {"train": self.train_total_loss_history,
                                          "valid": self.valid_total_loss_history}
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

            del total_loss, accuracy_total
        self.train_total_loss_histories += [self.train_total_loss_history]
        self.train_accuracy_histories += [self.train_accuracy_history]
        self.valid_total_loss_histories += [self.valid_total_loss_history]
        self.valid_accuracy_histories += [self.valid_accuracy_history]
        mean_total_losses_histories = {"train": np.mean(np.array(self.train_total_loss_histories), axis=0),
                                       "valid": np.mean(np.array(self.valid_total_loss_histories), axis=0)}
        var_losses_histories = {"train": np.std(np.array(self.train_total_loss_histories), axis=0),
                                "valid": np.std(np.array(self.valid_total_loss_histories), axis=0)}
        mean_accuracies_histories = {"train": np.mean(np.array(self.train_accuracy_histories), axis=0),
                                     "valid": np.mean(np.array(self.valid_accuracy_histories), axis=0)}
        var_accuracies_histories = {"train": np.std(np.array(self.train_accuracy_histories), axis=0),
                                    "valid": np.std(np.array(self.valid_accuracy_histories), axis=0)}
        labels = {"train": self.labels_train, "valid": self.labels_test}
        plot_performance(loss_total=mean_total_losses_histories,
                         std_loss=var_losses_histories,
                         accuracy=mean_accuracies_histories,
                         std_accuracy=var_accuracies_histories,
                         labels=labels,
                         results_path=self.results_path + "/" + hparams_string + "/",
                         filename=self.dataset_name)

        self.save_model()