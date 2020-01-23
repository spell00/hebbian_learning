from __future__ import print_function

import torch
from utils.utils import remove_nans
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.generative.autoencoders.autoencoder import ae

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard
from utils.visual_evaluation import plot_histogram

import numpy as np

from scipy.misc import logsumexp

def xavier_init(m):
    s =  np.sqrt( 2. / (m.in_features + m.out_features) )
    m.weight.data.normal_(0, s)

class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()

    def forward(self, h, g):
        return h * g



class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mom = 0
        self.lr = 0
    def set_layers(self, input_size, z1_size):
        # encoder: q(z | x)
        self.zs = []
        self.zs_test = []
        self.input_size = input_size
        self.z1_size = z1_size
        self.q_z_layers_pre = nn.ModuleList()
        self.q_z_layers_gate = nn.ModuleList()

        self.q_z_layers_pre.append( nn.Linear(np.prod(self.input_size), 300) )
        self.q_z_layers_gate.append( nn.Linear(np.prod(self.input_size), 300) )

        self.q_z_layers_pre.append( nn.Linear(300, 300) )
        self.q_z_layers_gate.append( nn.Linear(300, 300) )

        self.q_z_mean = nn.Linear(300, self.z1_size)
        self.q_z_logvar = nn.Linear(300, self.z1_size)

        # decoder: p(x | z)
        self.p_x_layers_pre = nn.ModuleList()
        self.p_x_layers_gate = nn.ModuleList()

        self.p_x_layers_pre.append( nn.Linear(self.z1_size, 300) )
        self.p_x_layers_gate.append( nn.Linear(self.z1_size, 300) )

        self.p_x_layers_pre.append( nn.Linear(300, 300) )
        self.p_x_layers_gate.append( nn.Linear(300, 300) )

        self.p_x_mean = nn.Linear(300, np.prod(self.input_size))

        self.sigmoid = nn.Sigmoid()

        self.Gate = Gate()

        # Xavier initialization (normal)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m)


    # AUXILIARY METHODS
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def calculate_likelihood(self, X, mode='test', S=5000):
        # set auxiliary variables for number of training and test sets
        N_test = X.size(0)

        # init list
        likelihood_test = []

        MB = 500

        if S <= MB:
            R = 1
        else:
            R = S / MB
            S = MB

        for j in range(N_test):
            if j % 100 == 0:
                print('{:.2f}%'.format(j / (1. * N_test) * 100))
            # Take x*
            x_single = X[j].unsqueeze(0)

            a = []
            for r in range(0, int(R)):
                # Repeat it for all training points
                x = x_single.expand(S, x_single.size(1))

                # pass through VAE
                x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)
                x_mean = x_mean.double()
                # RE
                RE = log_Bernoulli(x, x_mean, dim=1)

                # KL
                log_p_z = log_Normal_standard(z_q, dim=1)
                log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
                KL = -(log_p_z - log_q_z)

                a_tmp = (RE - KL)

                a.append( a_tmp.cpu().data.numpy() )

            # calculate max
            a = np.asarray(a)
            a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
            likelihood_x = logsumexp( a )
            likelihood_test.append(likelihood_x - np.log(len(a)))

        likelihood_test = np.array(likelihood_test)

        plot_histogram(-likelihood_test, self.model_history_path, mode)

        return -np.mean(likelihood_test)

    def calculate_lower_bound(self, X_full):
        # CALCULATE LOWER BOUND:
        lower_bound = 0.
        RE_all = 0.
        KL_all = 0.

        MB = 500

        for i in range(int(X_full.size(0) / MB)):
            x = X_full[i * MB: (i + 1) * MB].view(-1, np.prod(self.input_size))

            # pass through VAE
            x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)
            x_mean = x_mean.double()
            # RE
            RE = log_Bernoulli( x, x_mean )

            # KL
            log_p_z = log_Normal_standard( z_q, dim=1)
            log_q_z = log_Normal_diag( z_q, z_q_mean, z_q_logvar, dim=1 )
            KL = - torch.sum( log_p_z - log_q_z )

            RE_all += RE.cpu().item()
            KL_all += KL.cpu().item()

            # CALCULATE LOWER-BOUND: RE + KL - ln(N)
            lower_bound += (-RE + KL).cpu().item()

        lower_bound = lower_bound / X_full.size(0)

        return lower_bound

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        h0_pre = self.q_z_layers_pre[0](x)
        h0_gate = self.sigmoid( self.q_z_layers_gate[0](x) )
        h0 = self.Gate( h0_pre, h0_gate )

        h1_pre = self.q_z_layers_pre[1](h0)
        h1_gate = self.sigmoid( self.q_z_layers_gate[1](h0) )
        h1 = self.Gate( h1_pre, h1_gate )

        z_q_mean = self.q_z_mean(h1)
        z_q_logvar = self.q_z_logvar(h1)
        return z_q_mean, z_q_logvar

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_x(self, z):
        h0_pre = self.p_x_layers_pre[0](z)
        h0_gate = self.sigmoid( self.p_x_layers_gate[0](z) )
        h0 = self.Gate( h0_pre, h0_gate )

        h1_pre = self.p_x_layers_pre[1](h0)
        h1_gate = self.sigmoid( self.p_x_layers_gate[1](h0) )
        h1 = self.Gate( h1_pre, h1_gate )

        x_mean = self.sigmoid( self.p_x_mean(h1) )
        x_logvar = 0.
        return x_mean, x_logvar

    # THE MODEL: FORWARD PASS
    def forward(self, x, training=True, plot_z=True):
        # z ~ q(z | x)
        x = remove_nans(x)
        z_q_mean, z_q_logvar = self.q_z(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)
        if plot_z and training:
            self.zs.append(z_q.cpu().detach().numpy().tolist())
        elif plot_z and not training:
            self.zs_test.append(z_q.cpu().detach().numpy().tolist())

        # x_mean = p(x|z)
        x_mean, x_logvar = self.p_x(z_q)
        return x_mean, x_logvar, z_q, z_q_mean, z_q_logvar



