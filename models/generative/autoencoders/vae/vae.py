import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from models.generative.autoencoders.autoencoder import AE

from models.semi_supervised.deep_generative_models.layers.stochastic import GaussianSample, GumbelSoftmax, \
                                                                                    gaussian_mixture, swiss_roll
from utils.distributions import log_gaussian, log_standard_gaussian
from models.semi_supervised.deep_generative_models.layers.flow import NormalizingFlows, HouseholderFlow, ccLinIAF, SylvesterFlows
from models.generative.autoencoders.vae.GatedConv import GatedConv2d, GatedConvTranspose2d
import numpy as np


def onehot(label, k):
    y = torch.zeros(k)
    if label < k:
        y[label] = 1
    del label
    return y


class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()

    def forward(self, h, g):
        return h * g


class Perceptron(nn.Module):
    def __init__(self, dims, activation_fn=F.relu, output_activation=None):
        super(Perceptron, self).__init__()
        self.dims = dims
        self.activation_fn = activation_fn
        self.output_activation = output_activation

        self.layers = nn.ModuleList(list(map(lambda d: nn.Linear(*d), list(zip(dims, dims[1:])))))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers)-1 and self.output_activation is not None:
                x = self.output_activation(x)
            else:
                x = self.activation_fn(x)

        return x


class Encoder(nn.Module):
    def __init__(self, input_size, h_dim, z_dim, num_classes, y_dim, sample_layer=GaussianSample, a_dim=0):
        """
        Inference network

        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(Encoder, self).__init__()

        self.num_classes = num_classes
        self.y_dim = y_dim
        neurons = [input_size + y_dim + a_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        gates_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        bn = [nn.BatchNorm1d(neurons[i]) for i in range(1, len(neurons))]
        self.num_classes = num_classes
        self.a_dim = a_dim
        self.hidden = nn.ModuleList(linear_layers)
        self.gates_layers = nn.ModuleList(gates_layers)
        self.bn = nn.ModuleList(bn)
        try:
            self.sample = sample_layer(h_dim[-1], z_dim)
        except:
            self.sample = sample_layer
        self.cuda()
        self.Gate = Gate()

    def forward(self, x, y=torch.Tensor([]).cuda(), a=torch.Tensor([]).cuda(), input_shape=None):
        if input_shape is not None:
            x = x.view(-1, np.prod(input_shape)).float()
        x = torch.cat([x, y, a], dim=1)
        for layer, gate_layer, bn in zip(self.hidden, self.gates_layers, self.bn):
            x_pre = layer(x)
            x = gate_layer(x)
            x = bn(x)
            x = F.relu(x)
            x = self.Gate(x_pre, x)
            x = F.dropout(x)
        z = x
        try:
            sample1 = self.sample(x.shape[0], ndim=2, num_labels=self.num_classes)
        except:
            sample1 = self.sample(x)
        return sample1, z


class ConvEncoder(nn.Module):

    def __init__(self, h_dim, z_dim, planes, kernels, padding, pooling_layers,
                 sample_layer=GaussianSample,
                 y_size=0, a_size=0):
        """
        Inference network

        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(ConvEncoder, self).__init__()

        neurons = [planes[-1] + y_size + a_size, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        bn = [nn.BatchNorm1d(neurons[i]) for i in range(1, len(neurons))]
        conv_bn = [nn.BatchNorm2d(planes[i]) for i in range(1, len(planes))]
        conv_layers = [GatedConv2d(planes[i-1], planes[i], kernels[i-1], stride=1, padding=padding[i-1])
                       for i in range(1, len(planes))]

        self.hidden = nn.ModuleList(linear_layers)
        self.bn = nn.ModuleList(bn)
        self.conv_bn = nn.ModuleList(conv_bn)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.sample = sample_layer(h_dim[-1], z_dim)
        self.pool = nn.MaxPool2d(2, 2)
        self.cuda()
        self.pooling_layers = pooling_layers
        #self.reconstruct = nn.Linear(, 852)

    def forward(self, x, input_shape, y=torch.Tensor([]).cuda(), a=torch.Tensor([]).cuda()):
        if len(x.shape) == 2 and len(input_shape) == 3:
            x = x.view(-1, input_shape[0], input_shape[1], input_shape[2])
        for i, (conv_layer, conv_bn) in enumerate(zip(self.conv_layers, self.conv_bn)):
            x = conv_layer(x)
            x = F.relu(x)
            if self.pooling_layers[i]:
                x = self.pool(x)
            x = conv_bn(x)
            x = F.dropout2d(x, 0.3)

        x = x.squeeze()
        x = torch.cat([x, y, a], dim=1)
        for i, (layer, bn) in enumerate(zip(self.hidden, self.bn)):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout2d(x)
        z = x
        #self.reconstruct(z)
        return self.sample(x), z


class Decoder(nn.Module):
    def __init__(self, z_dim, h_dim, input_size, num_classes=0, a_dim=0):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(Decoder, self).__init__()
        neurons = [z_dim + num_classes + a_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        gates_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        bn = [nn.BatchNorm1d(neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)

        self.bn = nn.ModuleList(bn)
        self.gates_layers = nn.ModuleList(gates_layers)

        self.reconstruction = nn.Linear(int(h_dim[-1]), int(input_size))

        self.output_activation = nn.Tanh()
        self.Gate = Gate()

    def forward(self, z, y=torch.Tensor([])):
        y = y.cuda()
        if len(y) > 0:
            z = torch.cat([z, y], dim=1)
        for layer, gate_layer, bn in zip(self.hidden, self.gates_layers, self.bn):
            z_pre = layer(z)
            z = gate_layer(z)
            z = bn(z)
            z = F.relu(z)
            z = self.Gate(z_pre, z)
            z = F.dropout(z)

        return self.output_activation(self.reconstruction(z))

class ConvDecoder(nn.Module):
    def __init__(self, z_dim, h_dim, input_shape, planes, kernels, padding, unpooling_layers,
                 num_classes=0, a_dim=0):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(ConvDecoder, self).__init__()

        neurons = [planes[-1] + num_classes + a_dim, *h_dim]
        planes = list(reversed(planes))
        deconv_layers = [GatedConvTranspose2d(planes[i-1], planes[i], kernels[i-1], padding=padding[i-1], stride=1, activation=F.relu)
                         for i in range(1, len(planes))]
        linear_layers = [nn.Linear(z_dim + num_classes + a_dim, neurons[i]) for i in range(1, len(neurons))]
        deconv_bn = [nn.BatchNorm2d(planes[i]) for i in range(1, len(planes))]
        bn = [nn.BatchNorm1d(neurons[i]) for i in range(1, len(neurons))]

        self.linear_pre_deconv = nn.Linear(z_dim + num_classes + a_dim, planes[0])

        self.unpool = [nn.UpsamplingBilinear2d(2**i) for i in range(1, len(planes))]
        self.hidden = nn.ModuleList(linear_layers)
        self.deconv = nn.ModuleList(deconv_layers)
        self.deconv_bn = nn.ModuleList(deconv_bn)
        self.bn = nn.ModuleList(bn)
        self.planes = planes
        self.input_shape = input_shape

        self.a_dim = a_dim
        self.num_classes = num_classes

        self.output_activation = nn.Tanh()
        self.unpooling_layers = unpooling_layers
        self.reconstruction = nn.Linear(4096, int(np.prod(self.input_shape)))
        self.reconstruction_bn = nn.BatchNorm1d(4096)
    def forward(self, x, y=torch.Tensor([]).cuda(), a=torch.Tensor([]).cuda()):
        x = torch.cat([x, y], dim=1)
        for i, (layer, bn) in enumerate(zip(self.hidden, self.bn)):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x)

        #x = self.linear_pre_deconv(x)
        x = x.unsqueeze(2).unsqueeze(3)
        for i, (deconv, deconv_bn) in enumerate(zip(self.deconv, self.deconv_bn)):
            x = deconv(x)
            x = deconv_bn(x)
            x = F.relu(x)
            if self.unpooling_layers[i]:
                x = self.unpool[i](x)
            x = F.dropout2d(x, 0.3)

        x = x.view(int(x.shape[0]), -1)
        x = self.reconstruction_bn(x)
        x = self.reconstruction(x)
        #x = nn.functional.upsample_bilinear(x, size=int(self.input_shape[1]))
        #x = x.view(-1, np.prod(self.input_shape))
        return self.output_activation(x)


class VariationalAutoencoder(AE):
    def __init__(self, flavour, z_dims, h_dims, planes=None, pooling_layers=None, kernels=None, padding=None, auxiliary=False, n_flows=0,
                 a_dim=0, dropout=0.5, num_elements=None, hebb_layers=False):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encode/decoder pair for which
        a variational distribution is fitted to the
        encode. Also known as the M1 model in [Kingma 2014].

        :param dims: x, z and hidden dimensions of the networks
        """
        super(VariationalAutoencoder, self).__init__()
        print("a_dim (making sure it stays ok for ssl_vae)", a_dim)
        self.sylvester_flow = False
        self.dropout = dropout
        self.flow_type = "Vanilla"
        self._set_flavour(flavour)
        self.h_dims = h_dims
        self.a_dim = a_dim
        self.a_dim = a_dim
        self.num_elements = num_elements
        self.planes = planes
        self.kernels = kernels
        self.padding = padding
        self.pooling_layers = pooling_layers
        self.hebb_layers = hebb_layers
        if type(z_dims) is not list:
            self.z_dim_last = z_dims
            self.z_dims = z_dims = [z_dims]
        else:
            self.z_dims = z_dims = z_dims
            self.z_dim_last = z_dims[-1]
        if not self.ladder:
            self.z_dims = [self.z_dims[-1]]
        self.encoder = None
        self.ladder = False
        self.decoder = None
        self.flavour = flavour
        self.kl_divergence = 0
        if n_flows > 0 and not self.sylvester_flow and self.flow_flavour is not None:
            self.n_flows = n_flows
            self.add_flow(self.flow_flavour(in_features=[z_dims[-1]], n_flows=n_flows, h_last_dim=h_dims[-1],
                                            auxiliary=False))
        elif self.sylvester_flow and not self.ladder and self.flow_flavour is not None:
            self.add_flow(self.flow_flavour(in_features=[z_dims[-1]], n_flows=n_flows, h_last_dim=h_dims[-1],
                                            auxiliary=auxiliary, flow_flavour=flavour))

        self.cuda()

    def _set_flavour(self, flavour):
        if flavour == "Vanilla" or flavour == "vanilla":
            self.flow_flavour = None
            self.flow_type = "Vanilla"
            self.n_flows = 0
        # Normalizing flows
        elif flavour == "nf":
            self.flow_flavour = NormalizingFlows
            self.flow_type = "nf"
        # Householder flows
        elif flavour == "hf":
            self.flow_flavour = HouseholderFlow
            self.flow_type = "hf"
        # convex combination Linear Inverse Autoregressive Flows
        elif flavour == "ccLinIAF":
            self.flow_flavour = ccLinIAF
            self.flow_type = "ccLinIAF"
        # Orthogonal Sylvester Flow
        elif flavour == "o-sylvester":
            self.flow_flavour = SylvesterFlows
            self.sylvester_flow = True
            self.flow_type = "o-sylvester"
        # Householder Sylvester Flow
        elif flavour == "h-sylvester":
            self.flow_flavour = SylvesterFlows
            self.sylvester_flow = True
            self.flow_type = "h-sylvester"
        # Triangular Sylvester Flow
        elif flavour == "t-sylvester":
            self.flow_flavour = SylvesterFlows
            self.sylvester_flow = True
            self.flow_type = "t-sylvester"

    def set_vae_layers(self, sample_layer=""):
        self.prior_dist = sample_layer
        if sample_layer == "gaussian_mixture":
            print("\n\nGaussian Mixture\n\n")
            sample_layer = gaussian_mixture
        elif sample_layer == "swiss_roll":
            print("\n\nSwiss Roll\n\n")
            sample_layer = swiss_roll
        else:
            print("isotropic gaussian")
            sample_layer = GaussianSample
            self.prior_dist = "gaussian"
        self.encoder = Encoder(self.input_size, self.h_dims, self.z_dim_last, num_classes=self.num_classes,
                               sample_layer=sample_layer, y_dim=0)
        self.decoder = Decoder(self.z_dim_last, list(reversed(self.h_dims)), self.input_size)
        self.kl_divergence = 0
        self.sampler_layer = sample_layer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.cuda()

    def set_cvae_layers(self, h_dims, z_dim, planes, kernels, padding, pooling_layers):
        self.encoder = ConvEncoder(h_dim=h_dims, z_dim=z_dim, planes=planes,
                                   kernels=kernels, padding=padding, pooling_layers=pooling_layers)
        self.decoder = ConvDecoder(z_dim, list(reversed(h_dims)), input_shape=self.input_shape,
                                   kernels=list(reversed(kernels)), padding=list(reversed(padding)),
                                   planes=list(reversed(planes)), unpooling_layers=pooling_layers)

        self.kl_divergence = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.cuda()

    def _kld(self, z, q_param,  i, h_last, p_param=None, sylvester_params=None, auxiliary=False):
        """
        Computes the KL-divergence of
        some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]

        :param z: sample from q-distribuion
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        if self.flow_type == "nf" and self.n_flows > 0:
            (mu, log_var) = q_param
            if not auxiliary:
                f_z, log_det_z = self.flow(z, i, False)
            else:
                f_z, log_det_z = self.flow_a(z, i, True)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        elif self.flow_type in ["hf", "ccLinIAF"] and self.n_flows > 0:
            (mu, log_var) = q_param
            if not auxiliary:
                f_z = self.flow(z, i, h_last, False)
            else:
                f_z = self.flow_a(z, i, h_last, True)
            qz = log_gaussian(z, mu, log_var)
            z = f_z
        elif self.flow_type in ["o-sylvester", "h-sylvester", "t-sylvester"] and self.n_flows > 0:
            mu, log_var, r1, r2, q_ortho, b = q_param
            if not auxiliary:
                f_z = self.flow(z, r1, r2, q_ortho, b, i, False)
            else:
                f_z = self.flow_a(z, r1, r2, q_ortho, b, i, True)
            qz = log_gaussian(z, mu, log_var)
            z = f_z
        else:
            (mu, log_var) = q_param
            qz = log_gaussian(z, mu, log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)

        kl = qz - pz
        return kl

    def add_flow(self, flow):
        self.flow = flow
        self.flow.cuda()

    def add_flow_auxiliary(self, flow):
        self.flow_a = flow
        self.flow_a.cuda()

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        :param x: input data
        :return: reconstructed input
        """
        if y is not None:
            q, h = self.encoder(x, y)
        else:
            q, h = self.encoder(x)
        try:
            z = q[0]
            self.kl_divergence = self._kld(z, q[1:], i=0, h_last=h)
        except:
            # Clearly a problem with computing the kl divergence
            z = np.transpose(q)
            q2 = torch.Tensor(np.mean(q, 0)).view(-1, 1), torch.Tensor(np.var(q, 0)).view(-1, 1)
            self.kl_divergence = -torch.sum(self._kld(z, q2, i=0, h_last=h))
            z = torch.Tensor(q).cuda()


        x_mu = self.decoder(z)

        return x_mu, z

    def sample(self, z, y=None):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)


class GumbelAutoencoder(nn.Module):
    def __init__(self, dims, n_samples=100):
        super(GumbelAutoencoder, self).__init__()

        [self.input_size, z_dim, h_dim] = dims
        self.z_dim = z_dim
        self.n_samples = n_samples

        self.encoder = Perceptron([self.input_size, *h_dim])
        self.sampler = GumbelSoftmax(h_dim[-1], z_dim, n_samples)
        self.decoder = Perceptron([z_dim, *reversed(h_dim), self.input_size], output_activation=F.sigmoid)

        self.kl_divergence = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, qz):
        k = Variable(torch.FloatTensor([self.z_dim]), requires_grad=True)
        kl = qz * (torch.log(qz + 1e-8) - torch.log(1.0/k))
        kl = kl.view(-1, self.n_samples, self.z_dim)
        return torch.sum(torch.sum(kl, dim=1), dim=1)

    def forward(self, x, y=None, tau=1):
        x = self.encoder(x)

        sample, z_q = self.sampler(x, tau)
        self.kl_divergence = self._kld(z_q)

        x_mu = self.decoder(sample)

        return x_mu, z_q

    def sample(self, z):
        return self.decoder(z)


