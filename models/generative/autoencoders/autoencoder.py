from __future__ import print_function
import time
import torch
from utils.utils import create_missing_folders
from models.NeuralNet import NeuralNet
from models.semi_supervised.utils.loss import mse_loss_function as calculate_losses
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.visual_evaluation import plot_histogram
from scipy.misc import logsumexp
import torchvision as tv
from torch.nn import functional as F
from utils.distributions import log_gaussian, log_standard_gaussian
from scipy.stats import norm
from models.semi_supervised.utils.utils import onehot_array
import torch.backends.cudnn as cudnn
if torch.cuda.is_available():
    cudnn.enabled = True
    device = torch.device('cuda:0')
else:
    cudnn.enabled = False
    device = torch.device('cpu')


def safe_log(z):
    import torch
    return torch.log(z + 1e-7)

def rename_model(model_name, warmup, z1_size):
    model_name = model_name
    if model_name == 'vae_HF':
        number_combination = 0
    elif model_name == 'vae_ccLinIAF':
        number_of_flows = 1

    if model_name == 'vae_HF':
        model_name = model_name + '(T_' + str(number_of_flows) + ')'
    elif model_name == 'vae_ccLinIAF':
        model_name = model_name + '(K_' + str(number_combination) + ')'

    model_name = model_name + '_wu(' + str(warmup) + ')' + '_z1_' + str(z1_size)

    return model_name


class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()

    def forward(self, h, g):
        return h * g


class AE(NeuralNet):

    def __init__(self):
        super(AE, self).__init__()
        self.prior_dist = None
        self.test_loader = None
        self.train_loader = None
        self.val_loader = None
        self.mom = None
        self.lr = None
        self.batch_size = None
        self.latent_dim = None
        self.epsilon_std = None
        self.model_file_name = None
        self.input_size = None
        self.warmup = None
        self.early_stopping = None
        self.number_combination = None
        self.z = []
        self.zs_train = []
        self.zs_test = []
        self.zs_train_targets = []
        self.zs_test_targets = []
        self.n_layers = None
        self.n_hidden = None
        self.input_size = None

        self.encoder_pre = None
        self.encoder_gate = None
        self.q_z_mean = None
        self.q_z_log_var = None

        # decoder: p(x | z)
        self.decoder_pre = None
        self.decoder_gate = None
        self.reconstruction = None

        self.sigmoid = None
        self.tanh = None

        self.Gate = None

        self.encoder_pre = None
        self.encoder_gate = None
        self.bn_encoder = None

        self.decoder_pre = None
        self.decoder_gate = None
        self.bn_decoder = None
        self.ladder = None
        self.flavour = None
        self.z_dim_last = None
        self.optim_type = None

        self.best_loss = -1

        self.reconstruction_function = nn.MSELoss(size_average=False, reduce=False)

    def define_configurations(self, flavour, early_stopping=100, warmup=100, ladder=True, z_dim=40, epsilon_std=1.0,
                              model_name="vae", init="glorot", optim_type="adam", auxiliary=True, ssl=True):

        self.ladder = ladder
        self.ssl = ssl
        self.auxiliary = auxiliary
        self.flavour = flavour
        self.epsilon_std = epsilon_std
        self.warmup = warmup
        self.early_stopping = early_stopping
        self.init = init
        self.z_dim_last = z_dim
        self.set_init(init)
        self.optim_type = optim_type
        self.set_optim(optim_type)

        print('create model')
        # importing model
        self.model_file_name = rename_model(model_name, warmup, z_dim)

        if self.has_cuda:
            self.cuda()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def run(self, epochs=10, clip_grad=0, gen_rate=10):
        #best_model = self
        best_loss = 100000.
        e = 0
        self.train_loss_history = []
        self.train_rec_history = []
        self.train_kl_history = []

        self.val_loss_history = []
        self.val_rec_history = []
        self.val_kl_history = []

        time_history = []
        self.optimizer = self.optimization(self.parameters(), lr=float(self.lr), weight_decay=0.0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True,
                                                                    cooldown=0, patience=100)

        if self.has_cuda:
            self.cuda()

        for epoch in range(self.epoch, epochs + 1):
            self.epoch += 1
            self.optimizer.zero_grad()
            self.zs_train = []
            self.zs_train_targets = []
            self.zs_test = []
            self.zs_test_targets = []
            time_start = time.time()
            train_loss_epoch, train_rec_epoch, train_kl_epoch = self.train_vae(epoch, clip_grad)

            val_loss_epoch, val_rec_epoch, val_kl_epoch = self.evaluate_vae(mode='validation')
            self.scheduler.step(val_loss_epoch)
            time_end = time.time()

            time_elapsed = time_end - time_start

            # appending history
            self.train_loss_history.append(train_loss_epoch)
            self.train_rec_history.append(train_rec_epoch)
            self.train_kl_history.append(train_kl_epoch)
            self.val_loss_history.append(val_loss_epoch)
            self.val_rec_history.append(val_rec_epoch)
            self.val_kl_history.append(val_kl_epoch)
            time_history.append(time_elapsed)

            # printing results
            print('Epoch: {}/{}, Time elapsed: {:.8f}s\n'
                  '* Train loss: {:.8f}   (re: {:.8f}, kl: {:.8f})\n'
                  'o Val.  loss: {:.8f}   (re: {:.8f}, kl: {:.8f})\n'
                  '--> Early stopping: {}/{} (BEST: {:.8f})\n'.format(
                self.epoch, epochs, time_elapsed,
                train_loss_epoch, train_rec_epoch, train_kl_epoch,
                val_loss_epoch, val_rec_epoch, val_kl_epoch,
                e, self.early_stopping, best_loss
            ))

            # early-stopping
            if val_loss_epoch < best_loss and epoch > self.warmup:
                e = 0
                best_loss = val_loss_epoch
                #self.save_model()

            else:
                e += 1
                if e > self.early_stopping:
                    break

            if epoch < self.warmup:
                e = 0
            del val_loss_epoch, val_rec_epoch, val_kl_epoch, train_loss_epoch, train_rec_epoch, train_kl_epoch
        del best_loss
        # FINAL EVALUATION
        self.test_loss, self.test_re, self.test_kl, self.test_log_likelihood, self.train_log_likelihood, \
            self.test_elbo, self.train_elbo = self.evaluate_vae(mode='valid', gen_rate=gen_rate)

        self.print_final()

    def train_vae(self, epoch, clip_grad):
        # set loss to 0
        train_loss = 0
        train_rec = 0
        train_kl = 0
        # set model in training mode
        self.train()

        # start training
        if self.warmup == 0:
            beta = 1.
        else:
            beta = 1. * (epoch - 1) / self.warmup
            if beta > 1.:
                beta = 1.
        print('beta: {}'.format(beta))
        print("Labelled",len(self.train_loader))
        data = None
        reconstruction = None
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.view(-1, self.input_size)
            if self.has_cuda:
                data, target = data.cuda(), target.cuda()

            loss, rec, kl, reconstruction, z_q = self.calculate_losses(data, beta=1., likelihood=F.mse_loss)
            if type(z_q) == dict:
                z_q = z_q[-1]

            self.zs_train += [z_q]
            self.zs_train_targets += [target]
            # backward pass
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
            else:
                pass

            self.optimizer.step()
            self.optimizer.zero_grad()
            # optimization
            train_loss += loss.item()

            train_rec += rec.item()
            train_kl += kl.item()

            del rec, kl, target, z_q, loss
        print("Generating images")
        # if self.epoch % gen_rate == 0:
        del reconstruction, data
        # TODO address this in a proper way
        try:
            print("Unlabelled:", len(self.train_loader_unlabelled))
            for batch_idx, (data, _) in enumerate(self.train_loader_unlabelled):
                data = data.view(-1, self.input_size)
                if self.has_cuda:
                    data = data.cuda()

                loss, rec, kl, reconstruction, z_q = self.calculate_losses(data, beta=1., likelihood=F.mse_loss)

                # backward pass
                loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                else:
                    pass

                self.optimizer.step()
                # optimization
                train_loss += loss.item()

                train_rec += rec.item()
                train_kl += kl.item()

                del rec, kl, data, reconstruction, loss, clip_grad
        except:
            pass

        # calculate final loss
        train_loss /= len(self.train_loader)  # loss function already averages over batch size
        train_rec /= len(self.train_loader)  # rec already averages over batch size
        train_kl /= len(self.train_loader)  # kl already averages over batch size
        return train_loss, train_rec, train_kl









    def evaluate_vae(self, mode="validation", calculate_likelihood=True, gen_rate=10):
        # set loss to 0
        data, reconstruction = None, None

        with torch.no_grad():
            print("EVALUATION!")
            log_likelihood_test, log_likelihood_train, elbo_test, elbo_train = None, None, None, None
            evaluate_loss = 0
            evaluate_rec = 0
            evaluate_kl = 0
            # set model to evaluation mode

            if mode == "validation":
                data_loader = self.valid_loader
            elif mode == "valid":
                data_loader = self.test_loader

            if torch.cuda.is_available:
                has_cuda = True

            # evaluate
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.view(-1, self.input_size)
                if self.has_cuda:
                    data, target = data.cuda(), target.cuda()

                loss, rec, kl, reconstruction, z_q = self.calculate_losses(data, beta=1., likelihood=F.mse_loss)
                if type(z_q) == dict:
                    z_q = z_q[-1]
                self.zs_test += [z_q]
                self.zs_test_targets += [target]

                evaluate_loss += loss.item()
                evaluate_rec += rec.item()
                evaluate_kl += kl.item()
                del loss, kl, target, rec, z_q

            if mode == 'valid':
                # load all data
                test_data = Variable(data_loader.dataset.parent_ds.test_data)
                test_data = test_data.view(-1, torch.prod(torch.Tensor(self.input_shape)))
                test_target = Variable(data_loader.dataset.parent_ds.test_labels)
                full_data = Variable(self.train_loader.dataset.train_data)


                if has_cuda:
                    test_data, test_target, full_data = test_data.cuda(), test_target.cuda(), full_data.cuda()

                full_data = full_data.data.cpu().float().cuda()
                test_data = test_data.data.cpu().float().cuda()
                #full_data = torch.bernoulli(full_data)
                #test_data = torch.bernoulli(test_data)

                full_data = Variable(full_data.double(), requires_grad=True)

                # VISUALIZATION: plot reconstructions
                (_, z_mean_recon, z_logvar_recon), _ = self.encoder(test_data)
                z_recon = self.reparameterize(z_mean_recon, z_logvar_recon)


                # VISUALIZATION: plot generations
                z_sample_rand = Variable(torch.normal(torch.from_numpy(np.zeros((25, self.z_dim_last))).float(), 1.))
                if has_cuda:
                    z_sample_rand = z_sample_rand.cuda()

                full_data = full_data.data.cpu().float().cuda()
                test_data = test_data.data.cpu().float().cuda()
                elbo_test, elbo_train = self.calculate_elbo(full_data, test_data)
                if calculate_likelihood:
                    elbo_test, elbo_train, log_likelihood_test, log_likelihood_train = \
                        self.calculate_likelihood(full_data, test_data)

            # calculate final loss
            evaluate_loss /= len(data_loader)  # loss function already averages over batch size
            evaluate_rec /= len(data_loader)  # rec already averages over batch size
            evaluate_kl /= len(data_loader)  # kl already averages over batch size
            self.generate_random()
            self.generate_uniform_gaussian_percentiles()
            self.display_reconstruction(data, reconstruction)
            del data, reconstruction
            if mode == 'valid':
                return evaluate_loss, evaluate_rec, evaluate_kl, log_likelihood_test, log_likelihood_train, elbo_test, elbo_train
            else:
                if evaluate_loss < self.best_loss or self.best_loss == -1:
                    print("BEST EVALUATION LOSS: SAVING MODEL")
                    #self.best_loss = evaluate_loss
                    #self.save_model()

                return evaluate_loss, evaluate_rec, evaluate_kl

    def calculate_losses(self, data, lambda1=0., lambda2=0., beta=1., likelihood=F.mse_loss):
        if self.ladder:
            ladder = "ladder"
        else:
            ladder = "not_ladder"
        self.images_path = self.results_path + "/images/examples/generative/" + ladder + "/" + self.flavour + "/"
        create_missing_folders(self.images_path)
        data = torch.tanh(data)
        if self.flow_type in ["o-sylvester", "t-sylvester", "h-sylvester"] and not self.ladder:
            z_q = {0: None, 1: None}
            reconstruction, mu, log_var, self.log_det_j, z_q[0], z_q[-1] = self.run_sylvester(data,
                                                                                              auxiliary=self.auxiliary)
            log_p_zk = log_standard_gaussian(z_q[-1])
            # ln q(z_0)  (not averaged)
            # mu, log_var, r1, r2, q, b = q_param_inverse
            log_q_z0 = log_gaussian(z_q[0], mu, log_var=log_var) - self.log_det_j
            # N E_q0[ ln q(z_0) - ln p(z_k) ]
            self.kl_divergence = log_q_z0 - log_p_zk
            del log_q_z0, log_p_zk
        else:
            reconstruction, z_q = self(data)

        kl = beta * self.kl_divergence

        likelihood = torch.sum(likelihood(reconstruction, data.float(), reduce=False), dim=-1)

        if self.ladder:
            params = torch.cat([x.view(-1) for x in self.reconstruction.parameters()])
        else:
            params = torch.cat([x.view(-1) for x in self.decoder.reconstruction.parameters()])

        l1_regularization = lambda1 * torch.norm(params, 1).cuda()
        l2_regularization = lambda2 * torch.norm(params, 2).cuda()
        try:
            assert l1_regularization >= 0. and l2_regularization >= 0.
        except:
            print(l1_regularization, l2_regularization)
        loss = torch.mean(likelihood + kl.cuda() + l1_regularization + l2_regularization)

        del data, params, l1_regularization, l2_regularization, lambda1, lambda2

        return loss, torch.mean(likelihood), torch.mean(kl), reconstruction, z_q

    def plot_z(self, generated, max=5000):
        pos = None
        i = None
        label = None

        try:
            zs = np.vstack(torch.stack(self.zs_train).detach().cpu().numpy())
        except:
            zs = np.vstack(np.vstack(torch.stack(self.zs_train).detach().cpu().numpy()))

        labs = np.argmax(np.vstack(torch.stack(self.zs_train_targets).detach().cpu().numpy()), 1)
        z_list1 = dict(zip(list(self.labels_set), [[]] * len(self.labels_set)))
        z_list2 = dict(zip(list(self.labels_set), [[]] * len(self.labels_set)))
        fig, ax = plt.subplots()  # create figure and axis
        for i, label in enumerate(labs):
            if zs[i, 0] >= 0:
                z_list1[label].append(np.log(1 + np.array(zs[i, 0])))
            else:
                z_list1[label].append(-np.log(1 + np.absolute(np.array(zs[i, 0]))))
            if zs[i, 1] >= 0:
                z_list2[label].append(np.log(1 + zs[i, 1]))
            else:
                z_list2[label].append(-np.log(1 + np.absolute(zs[i, 1])))
        for i, label in enumerate(list(self.labels_set)[:max]):
            pos = [i for i, x in enumerate(labs) if str(x) == str(label)]
            ax.scatter(zs[pos, 0], zs[pos,1], s=3, marker='.', label=label)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.tight_layout()
        fig.tight_layout()
        path = "/".join([self.results_path, "plots/vae_z_plots", self.flavour, self.dataset_name, generated, self.prior_dist, "train"]) + "/"
        print("Plotting Z at:", path)
        create_missing_folders(path)
        fig.savefig(path +  self.dataset_name + "_" + str(self.epoch) + '_lr'+str(self.lr) + '_bs'+str(self.batch_size)
                    + '_flow'+str(self.flavour) + ".png")
        plt.close(fig)
        del zs, labs, z_list1, z_list2, fig, ax, i, label, pos, handles, labels, path,


        try:
            zs = np.vstack(torch.stack(self.zs_test).detach().cpu().numpy())
        except:
            zs = np.vstack(np.vstack(torch.stack(self.zs_test).detach().cpu().numpy()))

        labs = np.argmax(np.vstack(torch.stack(self.zs_test_targets).detach().cpu().numpy()), 1)
        z_list1 = dict(zip(list(self.labels_set), [[]] * len(self.labels_set)))
        z_list2 = dict(zip(list(self.labels_set), [[]] * len(self.labels_set)))
        fig, ax = plt.subplots()  # create figure and axis
        for i, label in enumerate(labs):
            if zs[i, 0] >= 0:
                z_list1[label].append(np.log(1 + np.array(zs[i, 0])))
            else:
                z_list1[label].append(-np.log(1 + np.absolute(np.array(zs[i, 0]))))
            if zs[i, 1] >= 0:
                z_list2[label].append(np.log(1 + zs[i, 1]))
            else:
                z_list2[label].append(-np.log(1 + np.absolute(zs[i, 1])))
        for i, label in enumerate(list(self.labels_set)[:max]):
            pos = [i for i, x in enumerate(labs) if str(x) == str(label)]
            ax.scatter(zs[pos, 0], zs[pos, 1], s=3, marker='.', label=label)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.tight_layout()
        fig.tight_layout()
        path = "/".join([self.results_path, "plots/vae_z_plots", self.flavour, self.dataset_name, generated, "valid"]) + "/"
        print("Plotting Z at:", path)
        create_missing_folders(path)
        fig.savefig(path +  self.dataset_name + "_" + str(self.epoch) + '_lr'+str(self.lr) + '_bs'+str(self.batch_size)
                    + '_flow'+str(self.flavour) + ".png")
        plt.close(fig)
        del zs, labs, z_list1, z_list2, fig, ax, i, label, pos, handles, labels, path,

    def calculate_elbo(self, test_data, full_data):
        # CALCULATE lower-bound
        t_ll_s = time.time()
        elbo_test = self.calculate_lower_bound(test_data)
        t_ll_e = time.time()
        print('Lower-bound time: {:.2f}'.format(t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        elbo_train = self.calculate_lower_bound(full_data)
        t_ll_e = time.time()
        print('Lower-bound time: {:.2f}'.format(t_ll_e - t_ll_s))
        return elbo_test, elbo_train

    def load_ae(self, load_history=False):
        print("LOADING PREVIOUSLY TRAINED autoencoder")
        trained_vae = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.state_dict')
        if self.ssl:
            trained_classifier = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + 'classifier.state_dict')
        self.load_state_dict(trained_vae)
        if load_history:
            self.epoch = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.epoch')
            self.train_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_loss')
            self.train_rec_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_re')
            self.train_kl_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_kl')
            self.val_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.val_loss')
            self.val_rec_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.val_re')
            self.val_kl_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.val_kl')

        if self.ssl:
            self.classifier = self.load_state_dict(trained_classifier)
    def save_model(self):
        # SAVING
        print("MODEL SAVED AT LOCATION:", self.model_history_path)
        create_missing_folders(self.model_history_path)
        torch.save(self.state_dict(), self.model_history_path + self.flavour + "_" + self.model_file_name +'.state_dict')
        if self.ssl:
            torch.save(self.classifier.state_dict(),  self.model_history_path + self.flavour + "_" + self.model_file_name +'classifier.state_dict')
        torch.save(self.train_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_loss')
        torch.save(self.train_rec_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_re')
        torch.save(self.train_kl_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_kl')
        torch.save(self.val_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.val_loss')
        torch.save(self.val_rec_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.val_re')
        torch.save(self.val_kl_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.val_kl')
        torch.save(self.epoch, self.model_history_path + self.flavour + "_" + self.model_file_name + '.epoch')
        #torch.save(self.test_log_likelihood, self.model_history_path + self.flavour + '.test_log_likelihood')
        #torch.save(self.test_loss, self.model_history_path + self.flavour + '.test_loss')
        #torch.save(self.test_re, self.model_history_path + self.flavour + '.test_re')
        #torch.save(self.test_kl, self.model_history_path + self.flavour + '.test_kl')

        # TODO SAVE images of 100 generated images

    def print_final(self, calculate_likelihood=False):
        print('FINAL EVALUATION ON TEST SET\n'
              'ELBO (TEST): {:.2f}\n'
              'ELBO (TRAIN): {:.2f}\n'
              'Loss: {:.2f}\n'
              're: {:.2f}\n'
              'kl: {:.2f}'.format(self.test_elbo, self.train_elbo,self.test_loss, self.test_re, self.test_kl))

        if calculate_likelihood:
            print('FINAL EVALUATION ON TEST SET\n'
                  'LogL (TEST): {:.2f}\n'
                  'LogL (TRAIN): {:.2f}'.format(self.test_log_likelihood, self.train_log_likelihood))

        with open(self.model_history_path + 'vae_experiment_log.txt', 'a') as f:
            print('FINAL EVALUATION ON TEST SET\n'
                  'ELBO (TEST): {:.2f}\n'
                  'ELBO (TRAIN): {:.2f}\n'
                  'Loss: {:.2f}\n'
                  're: {:.2f}\n'
                  'kl: {:.2f}'.format(self.test_elbo, self.train_elbo, self.test_loss, self.test_re, self.test_kl), file=f)

            if calculate_likelihood:
                print('FINAL EVALUATION ON TEST SET\n'
                      'LogL (TEST): {:.2f}\n'
                      'LogL (TRAIN): {:.2f}'.format(self.test_log_likelihood, self.train_log_likelihood), file=f)

    def save_config(self):
        pass

    def calculate_likelihood(self, data, mode='valid', s=5000, display_rate=100):
        # set auxiliary variables for number of training and valid sets
        n_test = data.size(0)

        # init list
        likelihood = []

        mb = 500

        if s <= mb:
            r = 1
        else:
            r = s / mb
            s = mb

        for j in range(n_test):
            n = 100 * (j / (1. * n_test))
            if j % display_rate == 0:
                print("\revaluating likelihood:", j, "/", n_test, -np.mean(likelihood), end="", flush=True)
            # Take x*                    print("\rProgress: {:.2f}%".format(progress), end="", flush=True)
            x_single = data[j].unsqueeze(0).view(self.input_shape[0], self.input_size)
            a = []
            for _ in range(0, int(r)):
                # Repeat it for all training points
                x = x_single.expand(s, x_single.size(1))

                # pass through VAE
                if self.flavour in ["ccLinIAF", "hf", "vanilla", "normflow"]:
                    loss, rec, kl, _ = self.calculate_losses(x)
                elif self.flavour in ["o-sylvester", "h-sylvester", "t-sylvester"]:
                    reconstruction, z_mu, z_var, ldj, z0, zk = self(x)
                    loss, rec, kl = calculate_losses(reconstruction, x, z_mu, z_var, z0, zk, ldj)
                else:
                    print(self.flavour, "is not a flavour, quiting.")
                    exit()

                a.append(loss.cpu().data.numpy())

            # calculate max
            a = np.asarray(a)
            a = np.reshape(a, (a.shape[0], 1))
            likelihood_x = logsumexp(a)
            likelihood.append(likelihood_x - np.log(len(a)))

        likelihood = np.array(likelihood)

        plot_histogram(-likelihood, self.model_history_path, mode)

        return -np.mean(likelihood)

    def calculate_lower_bound(self, x_full):
        # CALCULATE LOWER BOUND:
        lower_bound = 0.
        loss = torch.Tensor([])
        mb = 500

        for i in range(int(x_full.size(0) / mb)):

            x = x_full[i * mb: (i + 1) * mb].view(-1, self.input_size)

            if self.flavour in ["ccLinIAF", "hf", "vanilla", "normflow"]:
                loss, _, _, _ = self.calculate_losses(x)
            elif self.flavour in ["o-sylvester", "h-sylvester", "t-sylvester"]:
                reconstruction, z_mu, z_var, ldj, z0, zk = self(x)
                loss, _, _ = calculate_losses(reconstruction, x, z_mu, z_var, z0, zk, ldj)
            else:
                print(self.flavour, "is not a flavour, quiting.")
                exit()

            # CALCULATE LOWER-BOUND: re + kl - ln(N)
            lower_bound += loss.cpu().item()

        lower_bound = lower_bound / x_full.size(0)

        return lower_bound

    def plot_z_stats(self, z, path, generate="generated", max=5000):
        fig, ax = plt.subplots()  # create figure and axis
        plt.boxplot(z)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.tight_layout()
        fig.tight_layout()
        path = "/".join([path, "plots/vae_z_stats", generate]) + "/"
        create_missing_folders(path)
        fig.savefig(path + self.flavour + "_" + str(self.epoch) + '_lr' + str(self.lr) + '_bs'+str(self.batch_size) + ".png")
        plt.close(fig)

        if z.shape[1] == 2:
            self.plot_z(generated=generate)
        del z, path, generate

    def generate_random(self, max=1000):
        self.eval()
        print("GENERATING RANDOM IMAGES autoencoder!")
        hparams_string = "/".join(["num_elements"+str(self.num_elements), "n_flows"+str(self.n_flows),
                                   "z_dim"+str(self.z_dim_last), "unsupervised", "lr"+str(self.lr),
                                   "ladder"+str(self.ladder), self.flavour, self.prior_dist])
        images_path = self.images_path + hparams_string + "/generated_random/" + self.prior_dist + "/"
        create_missing_folders(images_path)

        rand_z = torch.randn(self.batch_size, self.z_dim_last).cuda()
        self.plot_z_stats(rand_z.detach().cpu().numpy(), generate="/random_generated/" + self.prior_dist + "/", path=images_path, max=max)
        new_x = self.sample(rand_z)
        if len(self.input_shape) > 1:
            images = new_x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data
            images_grid = tv.utils.make_grid(images)
            print("Images location:", images_path)
            tv.utils.save_image(images_grid, images_path + str(self.epoch) + self.dataset_name + "generated.png")
            del images_grid, images
        del rand_z, new_x, images_path, hparams_string


    def generate_uniform_gaussian_percentiles(self, n=20, verbose=1, max=1000):
        self.eval()
        print("GENERATING gaussian percentiles IMAGES autoencoder!")

        xs_grid = torch.Tensor(np.vstack([np.linspace(norm.ppf(0.01), norm.ppf(0.99), n**2) for _ in range(self.z_dim_last)]).T)

        hparams_string = "/".join(["num_elements"+str(self.num_elements), "n_flows"+str(self.n_flows),
                                   "z_dim"+str(self.z_dim_last), "unsupervised", "lr"+str(self.lr),
                                   "ladder"+str(self.ladder), self.flavour])
        images_path = self.images_path + "/" + hparams_string + "/gaussian_percentiles/" + "/" + self.prior_dist + "/"
        if verbose > 0:
            print("GENERATING SS DGM IMAGES AT", images_path)

        print("image path:", images_path)
        create_missing_folders(images_path)

        self.plot_z_stats(xs_grid, generate="/ugp_generated/" + self.prior_dist + "/", path=images_path, max=max)
        grid = torch.Tensor(xs_grid).to(device)

        new_x = torch.stack([self.sample(g.view(1, -1)) for g in grid])
        if len(self.input_shape) > 1:
            images = new_x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data

            assert n == int(images.shape[0]) / n
            images_grid = tv.utils.make_grid(images, int(np.sqrt(images.shape[0])))

            create_missing_folders(images_path)
            tv.utils.save_image(images_grid, images_path + str(self.epoch) + self.dataset_name + "gaussian_uniform_generated.png")
            del images_grid, images, new_x, xs_grid
    """
    def generate_uniform_gaussian_percentiles(self, epoch=0, verbose=0, show_pca=0, show_lda=0, n=20, drop_na=False):
    zs_grid = torch.stack([torch.Tensor(np.vstack([np.linspace(norm.ppf(0.05), norm.ppf(0.95), n**2)
                                                   for _ in range(self.z_dim_last)]).T)
                           for _ in range(self.num_classes)])

    # I get much better results squeezing values with tanh

    hparams_string = "/".join(["num_elements"+str(self.num_elements), "n_flows"+str(self.n_flows),
                               "z_dim"+str(self.z_dim_last), "a_dim"+str(self.a_dim), "lr"+str(self.lr),
                               "ladder"+str(self.ladder), self.flavour, "n_labelled"+str(len(self.train_loader))])
    images_path = self.results_path + "/" + hparams_string + "/gaussian_percentiles/"
    if verbose > 0:
        print("GENERATING SS DGM IMAGES AT", images_path)

    x_mu = [self.sample(torch.Tensor(zs_grid[i]).cuda()) for i in range(self.num_classes)]

    # plot_z_stats(rand_z.detach().cpu().numpy(), generate="generated")
    labels_set_ints = list(range(len(self.labels_set)))
    if len(self.input_shape) > 1:
        images = torch.stack([x_mu[i].view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data
                  for i in range(len(x_mu))])
        images = images.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        images_grid = tv.utils.make_grid(images, n)
        create_missing_folders(images_path)
        tv.utils.save_image(images_grid, images_path + "/" + str(epoch) + "gaussian_percentiles_generated.png")

    """


    def display_reconstruction(self, data, reconstruction):
        self.eval()
        print("GENERATING RECONSTRUCTION IMAGES autoencoder!")
        hparams_string = "/".join(["num_elements"+str(self.num_elements), "n_flows"+str(self.n_flows),
                                   "z_dim"+str(self.z_dim_last), "unsupervised", "lr"+str(self.lr),
                                   "ladder"+str(self.ladder), self.flavour])
        x = data.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data
        x_grid = tv.utils.make_grid(x)
        x_recon = reconstruction.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data
        x_recon_grid = tv.utils.make_grid(x_recon)
        images_path = self.images_path + hparams_string + "/recon/" + self.prior_dist + "/"
        print("Images location:", images_path)

        create_missing_folders(images_path)
        tv.utils.save_image(x_grid, images_path + "original_" + str(self.epoch) + ".png")
        tv.utils.save_image(x_recon_grid, images_path + "reconstruction_example_" + str(self.epoch) + ".png")

    def forward(self, *args):
        print("Nothing going on in forward of autoencoder.py")
        exit()
