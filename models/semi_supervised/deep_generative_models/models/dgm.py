import torch.nn as nn
from torch.nn import init
from scipy.stats import norm

import torch
import torchvision as tv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.semi_supervised.utils.utils import onehot_array, onehot
from models.generative.autoencoders.vae.vae import Encoder, Decoder, ConvDecoder, ConvEncoder
from models.discriminative.artificial_neural_networks.MultiLayerPerceptron import MLP
from models.discriminative.artificial_neural_networks.ConvNet import ConvNet
from utils.utils import create_missing_folders
import pylab
from list_parameters import ladder
import pandas as pd
from itertools import cycle
from torch.autograd import Variable
from dimension_reduction.ordination import ordination2d
#from parameters import vae_flavour, ladder
import numpy as np
import torch.backends.cudnn as cudnn
if torch.cuda.is_available():
    cudnn.enabled = True
    device = torch.device('cuda:0')
else:
    cudnn.enabled = False
    device = torch.device('cpu')

vae_flavour = "o-sylvester"
if ladder:
    from models.generative.autoencoders.vae.ladder_vae import LadderVariationalAutoencoder as VAE
else:
    if vae_flavour == "o-sylvester":
        from models.generative.autoencoders.vae.sylvester_vae import SylvesterVAE as VAE
    elif vae_flavour == "h-sylvester":
        from models.generative.autoencoders.vae.sylvester_vae import SylvesterVAE as VAE
    elif vae_flavour == "t-sylvester":
        from models.generative.autoencoders.vae.sylvester_vae import SylvesterVAE as VAE
    else:
        from models.generative.autoencoders.vae.vae import VariationalAutoencoder as VAE

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


def plot_performance(loss_total, loss_labelled, loss_unlabelled, accuracy, labels, results_path,
                     filename="NoName", verbose=0):
    fig2, ax21 = plt.subplots()
    try:
        ax21.plot(loss_total["train"], 'b-', label='Train total loss:' + str(len(labels["train"])))  # plotting t, a separately
        ax21.plot(loss_total["valid"], 'g-', label='Valid total loss:' + str(len(labels["valid"])))  # plotting t, a separately
        ax21.plot(loss_labelled["train"], 'b-.', label='Train labelled loss:' + str(len(labels["train"])))  # plotting t, a separately
        ax21.plot(loss_labelled["valid"], 'g-.', label='Valid labelled loss:' + str(len(labels["valid"])))  # plotting t, a separately
        ax21.plot(loss_unlabelled["train"], 'b.', label='Train unlabelled loss:' + str(len(labels["train"])))  # plotting t, a separately
        ax21.plot(loss_unlabelled["valid"], 'g.', label='Valid unlabelled loss:' + str(len(labels["valid"])))  # plotting t, a separately
        #ax21.plot(values["valid"], 'r-', label='Test:' + str(len(labels["valid"])))  # plotting t, a separately
    except:
        ax21.plot(loss_total["train"], 'b-', label='Train total loss:')  # plotting t, a separately
        ax21.plot(loss_total["valid"], 'g-', label='Valid total loss:')  # plotting t, a separately
        ax21.plot(loss_labelled["train"], 'b-.', label='Train labelled loss:')  # plotting t, a separately
        ax21.plot(loss_labelled["valid"], 'g-.', label='Valid labelled loss:')  # plotting t, a separately
        ax21.plot(loss_unlabelled["train"], 'b.', label='Train unlabelled loss:')  # plotting t, a separately
        ax21.plot(loss_unlabelled["valid"], 'g.', label='Valid unlabelled loss:')  # plotting t, a separately

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
    ax22.plot(accuracy["train"], 'b--', label='Train')  # plotting t, a separately
    ax22.plot(accuracy["valid"], 'g--', label='Valid')  # plotting t, a separately
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



class DeepGenerativeModel(VAE):
    def __init__(self, flow_type, z_dims, h_dims, n_flows, a_dim, auxiliary, num_elements=None, n_h=4,
                 hebb_layers=False, dropout=0.5, gt_input=-100):
        """
        M2 code replication from the paper
        'Semi-Supervised Learning with Deep Generative Models'
        (Kingma 2014) in PyTorch.

        The "Generative semi-supervised model" is a probabilistic
        model that incorporates label information in both
        inference and generation.

        Initialise a new generative model
        :param dims: dimensions of x, y, z and hidden layers.
        """
        self.gt_input = gt_input
        self.a_dim = a_dim
        self.flow_type = flow_type
        self.dropout = dropout
        self.auxiliary = auxiliary
        self.n_h = n_h
        super(DeepGenerativeModel, self).__init__(flavour=flow_type, z_dims=z_dims, h_dims=h_dims, auxiliary=auxiliary,
                                                  n_flows=n_flows, num_elements=num_elements, a_dim=a_dim,
                                                  hebb_layers=hebb_layers)
        self.z_dim, self.h_dims = z_dims[-1], h_dims
        self.n_flows = n_flows
        self.num_elements = num_elements

        self.train_total_loss_history = []
        self.train_labelled_loss_history = []
        self.train_unlabelled_loss_history = []
        self.train_accuracy_history = []
        self.train_kld_history = []
        self.valid_total_loss_history = []
        self.valid_labelled_loss_history = []
        self.valid_unlabelled_loss_history = []
        self.valid_accuracy_history = []
        self.valid_kld_history = []
        self.hebb_input_values_history = []
        self.epoch = 0
        self.indices_names = None
        self.hebb_layers = hebb_layers
        self.use_conv = False

    def set_dgm_layers(self, input_shape, num_classes, is_hebb_layers=False, is_clamp=False, extra_class=False):
        import numpy as np
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.input_size = np.prod(input_shape)
        self.set_vae_layers()
        self.encoder = Encoder(input_size=self.input_size, h_dim=self.h_dims, z_dim=self.z_dim,
                               num_classes=self.num_classes, y_dim=self.num_classes)
        self.decoder = Decoder(self.z_dim, list(reversed(self.h_dims)), self.input_size, num_classes=self.num_classes)

        hs = [self.h_dims[0] for _ in range(self.n_h)]
        if self.indices_names is None:
            self.indices_names = list(range(self.input_size))
        # The extra_class is previously added; this would put a second extra-class
        self.classifier = MLP(self.input_size, self.input_shape, self.indices_names, hs, self.num_classes,
                              dropout=self.dropout, is_hebb_layers=is_hebb_layers, is_clamp=is_clamp, gt_input=self.gt_input,
                              extra_class=extra_class)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def set_conv_dgm_layers(self, hs_ae, hs_class, z_dim, planes_ae, kernels_ae, padding_ae, pooling_layers_ae,
                            planes_c, kernels_c, pooling_layers_c, input_shape=None):
        if input_shape is not None:
            self.input_shape = input_shape
        self.set_cvae_layers(hs_ae, z_dim, planes_ae, kernels_ae, padding_ae, pooling_layers_ae)
        self.encoder = ConvEncoder(h_dim=hs_class, z_dim=z_dim, planes=planes_ae, kernels=kernels_ae,
                                   padding=padding_ae, pooling_layers=pooling_layers_ae)
        self.decoder = ConvDecoder(z_dim=z_dim, num_classes=self.num_classes, h_dim=list(reversed(hs_ae)), input_shape=self.input_shape,
                                   planes=list(reversed(planes_ae)), kernels=list(reversed(kernels_ae)),
                                   padding=list(reversed(padding_ae)), unpooling_layers=pooling_layers_ae)

        self.classifier = ConvNet(self.input_size, hs_class, self.num_classes, planes=planes_c, kernels=kernels_c,
                                  pooling_layers=pooling_layers_c, a_dim=self.a_dim, extra_class=False,
                                  indices_names=list(range(self.input_size)))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def set_dgm_layers_pretrained(self):
        classes_tensor1 = init.xavier_uniform_(torch.zeros([self.encoder.hidden[0].weight.size(0), self.num_classes]))
        classes_tensor2 = init.xavier_uniform_(torch.zeros([self.decoder.hidden[0].weight.size(0), self.num_classes]))
        self.encoder.hidden[0].weight = nn.Parameter(torch.cat((self.encoder.hidden[0].weight, classes_tensor1.cuda()), dim=1))
        self.decoder.hidden[0].weight = nn.Parameter(torch.cat((self.decoder.hidden[0].weight, classes_tensor2), dim=1))
        self.cuda()


    def classify(self, x, valid_bool, input_pruning=True, is_balanced_relu=False, start_pruning=3):
        print("x", x.shape)
        print("valid_bool", valid_bool.shape)
        logits = self.classifier(x, input_pruning=True, valid_bool=valid_bool, is_balanced_relu=is_balanced_relu)
        return logits

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.float()

        x = self.decoder(z, y=y)
        return x

    def generate_random(self, epoch=0, verbose=0, show_pca=1, show_lda=1, n=40, drop_na=False, keep_images=True,
                        only_na=False):
        hparams_string = "/".join(["num_elements"+str(self.num_elements), "n_flows"+str(self.n_flows),
                                   "z_dim"+str(self.z_dim_last), "a_dim"+str(self.a_dim), "lr"+str(self.lr),
                                   "ladder"+str(self.ladder), self.flavour])
        images_path = self.results_path + "/" + hparams_string + "/random/"
        create_missing_folders(images_path)
        if verbose > 0:
            print("GENERATING IMAGES AT", images_path)
        self.eval()

        rand_z = Variable(torch.randn(n*self.num_classes, self.z_dim))

        if not only_na:
            y = torch.cat([torch.Tensor(onehot_array(n*[i], self.num_classes)) for i in range(self.num_classes)])
        else:
            y = torch.cat(torch.Tensor(onehot_array(n*[self.num_classes], self.num_classes)))

        rand_z, y = rand_z.cuda(), y.cuda()
        x_mu = self.sample(rand_z, y)

        # plot_z_stats(rand_z.detach().cpu().numpy(), generate="generated")

        if len(self.input_shape) > 1 and keep_images:
            images = x_mu.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data
            images_grid = tv.utils.make_grid(images, 20)
            tv.utils.save_image(images_grid, images_path + "/" + str(epoch) + "only_na:" + str(only_na) +
                                "_generated.png")
        colnames = [list(self.labels_set)[one_hot.cpu().numpy().tolist().index(1)] for one_hot in y]
        df = pd.DataFrame(x_mu.transpose(1, 0).detach().cpu().numpy(), columns=colnames)
        if drop_na:
            try:
                df = df.drop(["N/A"], axis=1)
            except:
                pass
        if show_pca != 0 and epoch % show_pca == 0 and epoch != 0:
            try:
                ordination2d(df, "pca", epoch=self.epoch, images_folder_path=images_path, dataset_name=self.dataset_name, a=0.5,
                     verbose=0, info="generated")
            except:
                print("No pca.")
        if show_lda != 0 and epoch % show_lda == 0 and epoch != 0:
            try:
                ordination2d(df, "lda", epoch=self.epoch, images_folder_path=images_path, dataset_name=self.dataset_name, a=0.5,
                     verbose=0, info="generated")
            except:
                print("NO lda")
        del df, colnames, images_grid, x_mu, rand_z, y

        return images

    def plot_z_stats(self, z, results_path, generate="generated"):
        fig, ax = plt.subplots()  # create figure and axis
        plt.boxplot(z)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.tight_layout()
        fig.tight_layout()
        path = "/".join([results_path, "plots/vae_z_stats"]) + "/"
        fig.savefig(path + self.flavour + "_" + str(self.epoch) + '_lr' + str(self.lr) + '_bs' + str(self.batch_size)
                    + "_" + generate + ".png")
        plt.close(fig)

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

        y = torch.stack([torch.Tensor(onehot_array(n**2*[i], self.num_classes)) for i in range(n)])
        x_mu = [self.sample(torch.Tensor(zs_grid[i]).cuda(), y[i]) for i in range(self.num_classes)]

        # plot_z_stats(rand_z.detach().cpu().numpy(), generate="generated")
        labels_set_ints = list(range(len(self.labels_set)))
        if len(self.input_shape) > 1:
            images = torch.stack([x_mu[i].view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data
                      for i in range(len(x_mu))])
            images = images.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            images_grid = tv.utils.make_grid(images, n)
            create_missing_folders(images_path)
            tv.utils.save_image(images_grid, images_path + "/" + str(epoch) + "gaussian_percentiles_generated.png")


    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def run(self, n_epochs, auxiliary, mc=1, iw=1, lambda1=0., lambda2=0., verbose=1, show_progress=0,
            show_pca_train=0, show_lda_train=1, show_pca_valid=0, show_pca_generated=0, clip_grad=0.00001, warmup_n=-1,
            is_input_pruning=False, start_pruning=-1, schedule=True, show_lda_generated=1, is_balanced_relu=False,
            limit_examples=1000, keep_history=True, decay=1e-8, alpha_rate=0.1, t_max=1, generate_extra_class=100):
        from models.semi_supervised.deep_generative_models.inference import SVI, DeterministicWarmup,\
            ImportanceWeightedSampler
        try:
            alpha = alpha_rate * len(self.train_loader_unlabelled) / len(self.train_loader)
        except:
            self.train_loader_unlabelled = self.train_loader
            alpha = alpha_rate * len(self.train_loader_unlabelled) / len(self.train_loader)

        if torch.cuda.is_available():
            self.cuda()
        hparams_string = "/".join(["num_elements"+str(self.num_elements), "n_flows"+str(self.n_flows),
                                   "z_dim"+str(self.z_dim_last), "a_dim"+str(self.a_dim), "lr"+str(self.lr),
                                   "ladder"+str(self.ladder), self.flavour])
        self.images_path = self.results_path + "/" + hparams_string
        self.valid_bool = [1.] * np.prod(self.input_shape)
        if warmup_n == -1:
            print("Warmup on: ", 4*len(self.train_loader_unlabelled)*100)
            beta = DeterministicWarmup(n=4*len(self.train_loader_unlabelled)*100, t_max=t_max)
        elif warmup_n > 0:
            print("Warmup on: ", warmup_n)
            beta = DeterministicWarmup(n=warmup_n, t_max=t_max)
        else:
            beta = 1.


        sampler = ImportanceWeightedSampler(mc, iw)

        elbo = SVI(self, beta=beta, labels_set=self.labels_set, images_path=self.images_path, dataset_name=self.dataset_name,
                   auxiliary=auxiliary, batch_size=self.batch_size, likelihood=F.mse_loss, sampler=sampler,
                   ladder=self.ladder)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, verbose=True, cooldown=0,
                                                               patience=20)

        best_loss = 100000
        early = 0
        best_accuracy = 0

        involment_df = pd.DataFrame(index=self.indices_names)
        print("Log file created: ",  "logs/" + self.__class__.__name__ + "_parameters.log")
        file_parameters = open("logs/" + self.__class__.__name__ + "_parameters.log", 'w+')
        #print("file:", file_parameters)
        print(*("LABELLED:", len(self.train_loader)), sep="\t", file=file_parameters)
        print("UNLABELLED:", len(self.train_loader_unlabelled), sep="\t", file=file_parameters)
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
        print("Unlabeled shape", len(self.train_loader_unlabelled))
        flag_na = False
        for epoch in range(self.epoch, n_epochs):

            file = open("logs/" + self.__class__.__name__ + ".log", 'a+')
            file_involvment = open("logs/" + self.__class__.__name__ + "_involvment.log", 'a+')
            self.epoch += 1
            #self.train()
            total_loss, labelled_loss, unlabelled_loss, accuracy, accuracy_total = (0, 0, 0, 0, 0)

            print("epoch", epoch, file=file)
            if verbose > 0:
                print("epoch", epoch)
            c = 0
            recs_train = None
            ys_train = None

            # https://docs.python.org/2/library/itertools.html#itertools.cycle
            # cycle make it so if train_loader_unlabelled is not finished, it will "cycle" (repeats) indefinitely

            for (x, y), (u, _) in zip(cycle(self.train_loader), self.train_loader_unlabelled):
                if not self.use_conv:
                    x = x.view(-1, np.prod(x.shape[1:]))
                optimizer.zero_grad()
                c += len(x)
                progress = 100 * c / len(self.train_loader_unlabelled) / self.batch_size

                if verbose > 1:
                    print("\rProgress: {:.2f}%".format(progress), end="", flush=True)

                # Wrap in variables
                x, y, u = Variable(x), Variable(y), Variable(u)

                if torch.cuda.is_available():
                    # They need to be on the same device and be synchronized.
                    x, y = x.cuda(device=0), y.cuda(device=0)
                    u = u.cuda(device=0)

                if self.epoch > generate_extra_class:
                    if flag_na:
                        print("\n\n\nStarting include generated images for n/a!\n\n\n")
                    else:
                        pass
                    x_na = self.generate_random(n=100, verbose=0, keep_images=False, only_na=True, epoch=self.epoch)
                    x_na = x_na.view(-1, np.prod(self.input_shape))
                    x = torch.cat([x, x_na], 0)
                    y = torch.cat([y, torch.stack([torch.Tensor(onehot(self.num_classes, self.num_classes))
                                                   for _ in range(x_na.shape[0])]).cuda()])

                L, rec = elbo(x.float(), y.float(), self.valid_bool)
                U, _ = elbo(u.float(), y=None, valid_bool=self.valid_bool)

                # Add auxiliary classification loss q(y|x)
                logits = self.classify(x, valid_bool=self.valid_bool, input_pruning=is_input_pruning,
                                       start_pruning=start_pruning)
                classification_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()
                params = torch.cat([x.view(-1) for x in self.parameters()])
                l1_regularization = lambda1 * torch.norm(params, 1)
                l2_regularization = lambda2 * torch.norm(params, 2)

                if np.isnan(L.item()):
                    print("Problem with the LABELED loss function in dgm.py. Setting the loss to 0")
                    L = Variable(torch.Tensor([0.]).to(device), requires_grad=False)

                if np.isnan(U.item()):
                    print("Problem with the UNLABELED loss function in dgm.py. Setting the loss to 0")
                    U = Variable(torch.Tensor([0.]).to(device), requires_grad=False)

                if np.isnan(l1_regularization.item()):
                    print("Problem with the l1 value in dgm.py. Setting to 0")
                    l1_regularization = Variable(torch.Tensor([0.]).to(device), requires_grad=False)

                if np.isnan(l2_regularization.item()):
                    print("Problem with the l2 value in dgm.py. Setting to 0")
                    l2_regularization = Variable(torch.Tensor([0.]).to(device), requires_grad=False)
                if np.isnan(classification_loss.item()):
                    print("Problem with the CLASSIFICATION loss function in dgm.py. Setting the loss to 0")
                    classification_loss = Variable(torch.Tensor([0.]).to(device), requires_grad=False)


                J_alpha = L - alpha * classification_loss + U + l1_regularization + l2_regularization

                J_alpha.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem.
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                else:
                    pass
                if np.isnan(J_alpha.item()):
                    print("loss is nan")

                total_loss += J_alpha.item()
                labelled_loss += L.item()
                unlabelled_loss += U.item()

                _, pred_idx = torch.max(logits, 1)
                _, lab_idx = torch.max(y, 1)
                accuracy_total += torch.mean((pred_idx.data[0] == lab_idx.data[0]).float())
                accuracy += torch.mean((pred_idx.data[0] == lab_idx.data[0]).float())
                if recs_train is None:
                    ys_train = y
                    recs_train = rec
                elif recs_train.shape[0] < limit_examples:
                    recs_train = torch.cat([recs_train, rec], dim=0)
                    ys_train = torch.cat([ys_train, y], dim=0)
                optimizer.step()
                hebb_round = 1
                del J_alpha, L, U, x, u, rec, l1_regularization, l2_regularization, pred_idx, lab_idx, classification_loss, _

            #if epoch % hebb_round == 0 and epoch != 0:
            #    if self.hebb_layers:
            #        fcs, self.valid_bool = self.classifier.hebb_layers.compute_hebb(total_loss, epoch,
            #                                    results_path=self.results_path, fcs=self.classifier.fcs, verbose=3)
            #        alive_inputs = sum(self.valid_bool)
            #        if alive_inputs < len(self.valid_bool):
            #            print("Current input size:", alive_inputs, "/", len(self.valid_bool))
#
            #        hebb_input_values = self.classifier.hebb_layers.hebb_input_values
             #       self.classifier.fcs = fcs

                    # The last positions are for the auxiliary network, if using auxiliary deep generative model
            #        involment_df = pd.concat((involment_df, pd.DataFrame(hebb_input_values.cpu().numpy()[:-self.a_dim],
             #                                                            index=self.indices_names)), axis=1)
            #        involment_df.columns = [str(a) for a in range(involment_df.shape[1])]
            #        last_col = str(int(involment_df.shape[1])-1)
            #        print("epoch", epoch, "last ", last_col, file=file_involvment)
            #        print(involment_df.sort_values(by=[last_col], ascending=False), file=file_involvment)


            #colnames = [list(self.labels_set)[one_hot.tolist().index(1)] for one_hot in y]
            #new_cols = colnames * iw * mc
            #dataframe = pd.DataFrame(recs_train.transpose(1, 0).detach().cpu().numpy(), columns=new_cols)

            #if show_pca_train > 0 and epoch % show_pca_train == 0 and epoch != 0:
            #    ordination2d(dataframe, "pca", epoch=self.epoch, images_folder_path=self.images_path,
            #                 dataset_name=self.dataset_name, a=0.5, verbose=0, info="train", show_images=show_pca_train)
            #if show_lda_train > 0 and epoch % show_lda_train == 0 and epoch != 0:
            #    ordination2d(dataframe, "lda", epoch=self.epoch, images_folder_path=self.images_path,
            #                 dataset_name=self.dataset_name, a=0.5, verbose=0, info="train", show_images=show_lda_train)
            m = len(self.train_loader_unlabelled)

            self.eval()

            if keep_history:
                self.train_total_loss_history += [(total_loss / m)]
                self.train_labelled_loss_history += [(labelled_loss / m)]
                self.train_unlabelled_loss_history += [(unlabelled_loss / m)]
                self.train_accuracy_history += [(accuracy / m)]
                self.train_kld_history += [(torch.sum(self.kl_divergence).item())]

            print("Epoch: {}".format(epoch), sep="\t", file=file)
            print("[Train]\t\t J_a: {:.2f}, L: {:.2f}, U: {:.2f}, "
                  "accuracy: {:.4f}, kld: {:.1f}".format(total_loss / m, labelled_loss / m, unlabelled_loss / m,
                                    accuracy_total / m, torch.sum(self.kl_divergence).item()), sep="\t", file=file)
            if verbose > 0:
                print("[Train]\t\t J_a: {:.2f}, L: {:.2f}, U: {:.2f}, accuracy: {:.4f}, kld: {:.1f}"
                      .format(total_loss / m, labelled_loss / m, unlabelled_loss / m, accuracy_total / m,
                              torch.sum(self.kl_divergence).item()))

            total_loss, labelled_loss, unlabelled_loss, accuracy, accuracy_total = (0, 0, 0, 0, 0)
            recs_valid = None
            ys_valid = None
            for x, y in self.valid_loader:
                x, y = Variable(x), Variable(y)

                if torch.cuda.is_available():
                    x, y = x.cuda(device=0), y.cuda(device=0)
                if not self.use_conv:
                    x = x.view(-1, np.prod(x.shape[1:]))

                L, rec = elbo(x.float(), y, self.valid_bool, valid=True)

                U, _ = elbo(x.float(), y=None, valid_bool=self.valid_bool, valid=True)

                logits = self.classify(x, valid_bool=self.valid_bool, input_pruning=is_input_pruning,
                                       start_pruning=start_pruning)
                classification_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()
                J_alpha = L - alpha * classification_loss + U # l1_regularization + l2_regularization

                total_loss += J_alpha.item()
                labelled_loss += L.item()
                unlabelled_loss += U.item()

                _, pred_idx = torch.max(logits, 1)
                _, lab_idx = torch.max(y, 1)
                accuracy_total += torch.mean((pred_idx.data[0] == lab_idx.data[0]).float())
                accuracy += torch.mean((pred_idx.data[0] == lab_idx.data[0]).float())

                if recs_valid is None:
                    ys_valid = y
                    recs_valid = rec
                elif recs_train.shape[0] < limit_examples:
                    recs_valid = torch.cat([recs_valid, rec], dim=0)
                    ys_valid = torch.cat([ys_valid, y], dim=0)

                del J_alpha, L, U, pred_idx, lab_idx

            #colnames = [list(self.labels_set)[one_hot.tolist().index(1)] for one_hot in y]
            #new_cols = colnames * iw * mc
            #dataframe = pd.DataFrame(recs_valid.transpose(1, 0).detach().cpu().numpy(), columns=new_cols)
            #if show_pca_valid > 0 and epoch % show_pca_valid == 0 and epoch != 0:
            #    ordination2d(dataframe, "pca", epoch=self.epoch, images_folder_path=self.images_path,
            #                 dataset_name=self.dataset_name, a=0.5, verbose=1, info="valid")
            m = len(self.valid_loader)
            print("[Validation]\t J_a: {:.2f}, L: {:.2f}, U: {:.2f}, accuracy: {:.4f} , kld: {:.1f}"
                  .format(total_loss / m,  labelled_loss / m, unlabelled_loss / m, accuracy / m,
                          torch.sum(self.kl_divergence)), sep="\t", file=file)
            if verbose > 0:
                print("[Validation]\t J_a: {:.2f}, L: {:.2f}, U: {:.2f}, accuracy: {:.4f} , kld: {:.1f}"
                      .format(total_loss / m, labelled_loss / m, unlabelled_loss / m, accuracy / m,
                              torch.sum(self.kl_divergence)))

            if keep_history:
                self.valid_total_loss_history += [(total_loss / m)]
                self.valid_labelled_loss_history += [(labelled_loss / m)]
                self.valid_unlabelled_loss_history += [(unlabelled_loss / m)]
                self.valid_accuracy_history += [(accuracy / m)]
                self.valid_kld_history += [(torch.sum(self.kl_divergence).item())]

            # early-stopping
            if (accuracy > best_accuracy or total_loss < best_loss) and epoch > self.warmup:
                print("BEST LOSS!", total_loss / m)
                early = 0
                best_loss = total_loss

                #self.save_model()

            else:
                early += 1
                if early > self.early_stopping:
                    break

            if epoch < self.warmup:
                print("Warmup:", 100 * epoch / self.warmup, "%", sep="\t", file=file)
                early = 0

            _ = self.generate_random(epoch, verbose=1, show_pca=show_pca_generated, show_lda=show_lda_generated)
            _ = self.generate_random(n=100, verbose=0, keep_images=True, only_na=True, epoch=self.epoch)

            self.display_reconstruction(epoch, x, rec)
            try:
                self.generate_uniform_gaussian_percentiles(epoch)
            except:
                print("Did not generate uniform gaussian")
            total_losses_histories = {"train": self.train_total_loss_history, "valid": self.valid_total_loss_history}
            labelled_losses_histories = {"train": self.train_labelled_loss_history, "valid": self.valid_labelled_loss_history}
            unlabelled_losses_histories = {"train": self.train_unlabelled_loss_history, "valid": self.valid_unlabelled_loss_history}
            accuracies_histories = {"train": self.train_accuracy_history, "valid": self.valid_accuracy_history}
            labels = {"train": self.labels_train, "valid": self.labels_test}
            if show_progress > 0 and epoch % show_progress == 0 and epoch != 0:
                plot_performance(loss_total=total_losses_histories,
                             loss_labelled=labelled_losses_histories,
                             loss_unlabelled=unlabelled_losses_histories,
                             accuracy=accuracies_histories,
                             labels=labels,
                             results_path=self.results_path + "/" + hparams_string + "/",
                             filename=self.dataset_name)
            if schedule:
                scheduler.step(total_loss)
            file.close()
            file_involvment.close()

            del total_loss, labelled_loss, unlabelled_loss, accuracy, self.kl_divergence, recs_train, \
                rec, ys_train, ys_valid, recs_valid, accuracies_histories

    def define_configurations(self, flavour, early_stopping=100, warmup=100, ladder=True, z_dim=40, epsilon_std=1.0,
                              model_name="vae", init="glorot", optim_type="adam"):

        self.flavour = flavour
        self.epsilon_std = epsilon_std
        self.warmup = warmup
        self.early_stopping = early_stopping

        # importing model
        self.model_file_name = rename_model(model_name, warmup, z_dim)

        if self.has_cuda:
            self.cuda()

    def forward(self, x, y=None):
        # Add label and data and generate latent variable
        if len(x.shape) < len(self.input_shape):
            x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        (z, z_mu, z_log_var), h = self.encoder(x, y=y)
        self.kl_divergence = self._kld(z, (z_mu, z_log_var), i=0, h_last=h)
        # Reconstruct data point from latent data and label
        x_mu = self.decoder(z, y=y)

        return x_mu

    def load_model(self):
        print("LOADING PREVIOUSLY TRAINED VAE and classifier")
        trained_vae = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.state_dict')
        trained_classifier = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + 'classifier.state_dict')
        self.load_state_dict(trained_vae)
        self.classifier.load_state_dict(trained_classifier)
        self.epoch = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.epoch')
        self.train_total_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_total_loss')
        self.train_labelled_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_labelled_loss')
        self.train_unlabelled_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_unlabelled_loss')
        self.train_accuracy_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_accuracy')
        self.train_kld_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_kld')
        self.valid_total_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_total_loss')
        self.valid_labelled_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_labelled_loss')
        self.valid_unlabelled_loss_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_unlabelled_loss')
        self.valid_accuracy_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_accuracy')
        self.valid_kld_history = torch.load(self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_kld')

    def save_model(self):
        # SAVING
        print("MODEL (with classifier) SAVED AT LOCATION:", self.model_history_path)
        create_missing_folders(self.model_history_path)
        torch.save(self.state_dict(), self.model_history_path + self.flavour + "_" + self.model_file_name +'.state_dict')
        torch.save(self.classifier.state_dict(), self.model_history_path + self.flavour + "_" + self.model_file_name +'classifier.state_dict')
        torch.save(self.train_total_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_total_loss')
        torch.save(self.train_labelled_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_labelled_loss')
        torch.save(self.train_unlabelled_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_unlabelled_loss')
        torch.save(self.train_accuracy_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_accuracy')
        torch.save(self.train_kld_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.train_kld')
        torch.save(self.valid_total_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_total_loss')
        torch.save(self.valid_labelled_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_labelled_loss')
        torch.save(self.valid_unlabelled_loss_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_unlabelled_loss')
        torch.save(self.valid_accuracy_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_accuracy')
        torch.save(self.valid_kld_history, self.model_history_path + self.flavour + "_" + self.model_file_name + '.valid_kld')
        torch.save(self.epoch, self.model_history_path + self.flavour + "_" + self.model_file_name + '.epoch')
        #torch.save(self.test_log_likelihood, self.model_history_path + self.flavour + '.test_log_likelihood')
        #torch.save(self.test_loss, self.model_history_path + self.flavour + '.test_loss')
        #torch.save(self.test_re, self.model_history_path + self.flavour + '.test_re')
        #torch.save(self.test_kl, self.model_history_path + self.flavour + '.test_kl')








    def display_reconstruction(self, epoch, data, reconstruction, display_rate=1):
        hparams_string = "/".join(["num_elements"+str(self.num_elements), "n_flows"+str(self.n_flows),
                                   "z_dim"+str(self.z_dim_last), "a_dim"+str(self.a_dim), "lr"+str(self.lr),
                                   "ladder"+str(self.ladder), self.flavour])
        images_path = self.results_path + "/" + hparams_string + "/reconstruction/"
        create_missing_folders(images_path)
        x = data.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]).data
        x_grid = tv.utils.make_grid(x)
        x_recon = reconstruction.view(-1, self.input_shape[0], self.input_shape[1],
                                      self.input_shape[2]).data
        x_recon_grid = tv.utils.make_grid(x_recon)

        if epoch % display_rate == 0:
            print("GENERATING RECONSTRUCTION IMAGES autoencoder!")
            tv.utils.save_image(x_grid, images_path + str(epoch) + "_original.png")
            tv.utils.save_image(x_recon_grid, images_path + str(epoch) + "_reconstruction_example.png")













