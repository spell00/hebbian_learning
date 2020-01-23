import torch.nn.functional as F
import torch
from torch.autograd import Variable
import pandas as pd
from utils.plot_performance import plot_performance
import numpy as np
def run(self, n_epochs, verbose=1, clip_grad=0, is_input_pruning=False, start_pruning=3, show_progress=20,
        is_balanced_relu=True, plot_progress=20, hist_epoch=20, all0=False, overall_mean=False):
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
        total_loss, accuracy, accuracy_total = (0, 0, 0)

        print("epoch", self.epoch, file=file)
        if verbose > 0:
            print("epoch", self.epoch)
        c = 0
        for i, (x, y) in enumerate(self.train_loader):
            c += len(x)
            # progress = 100 * c / len(self.train_loader) / self.batch_size
            # print("Progress: {:.2f}%".format(progress))

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

        if self.epoch % hebb_round == 0 and self.epoch != 0:
            if self.is_hebb_layers:
                self.fcs, self.valid_bool = self.hebb_layers.compute_hebb(total_loss, self.epoch,
                                                                          results_path=self.results_path, fcs=self.fcs,
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
                accuracy_total += torch.mean((pred_idx.data[0] == lab_idx.data[0]).float())
                accuracy += torch.mean((pred_idx.data[0] == lab_idx.data[0]).float())

            except:
                lab_idx = y
                accuracy_total += torch.mean((pred_idx.data[0] == lab_idx.data).float())
                accuracy += torch.mean((pred_idx.data[0] == lab_idx.data).float())

            optimizer.step()
            optimizer.zero_grad()

            del loss, x, y

        m = len(self.valid_loader)
        print("[Validation]\t J_a: {:.2f}, accuracy: {:.4f}".format(total_loss / m,
                                                                    accuracy / m), sep="\t", file=file)
        if verbose > 0:
            print("[Validation]\t J_a: {:.2f}, accuracy: {:.4f}".format(total_loss / m, accuracy / m))
        # m = len(self.test_loader)

        if self.epoch % plot_progress == 0:
            self.valid_total_loss_history += [(total_loss / m)]
            self.valid_accuracy_history += [(accuracy / m)]

        # early-stopping
        if (accuracy > best_accuracy or total_loss < best_loss):
            # print("BEST LOSS!", total_loss / m)
            early = 0
            best_loss = total_loss
            # self.save_model()

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
