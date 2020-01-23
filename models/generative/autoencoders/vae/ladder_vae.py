import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from models.generative.autoencoders.vae.vae import Decoder
from utils.distributions import log_gaussian, log_standard_gaussian
from models.semi_supervised.deep_generative_models.layers.stochastic import GaussianSample, GaussianMerge
import torch
from torch.autograd import Variable

class LadderEncoder(nn.Module):
    def __init__(self, input_size, h_dim, z_dim):
        """
        The ladder encode differs from the standard encode
        by using batch-normalization and LReLU activation.
        Additionally, it also returns the transformation x.

        :param dims: dimensions [input_dim, [hidden_dims], [latent_dims]].
        """
        super(LadderEncoder, self).__init__()
        self.z_dim = z_dim
        self.in_features = input_size
        self.out_features = h_dim

        self.linear = nn.Linear(input_size, h_dim)
        self.batchnorm = nn.BatchNorm1d(h_dim)
        self.sample = GaussianSample(h_dim, z_dim)

    def forward(self, x):
        x = self.linear(x)
        x = F.leaky_relu(x, 0.1)
        x = self.batchnorm(x)
        x = F.dropout(x)
        return self.sample(x), x


class LadderDecoder(nn.Module):
    def __init__(self, z_dim, h_dim, input_size):
        """
        The ladder dencoder differs from the standard encode
        by using batch-normalization and LReLU activation.
        Additionally, it also returns the transformation x.

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(LadderDecoder, self).__init__()

        self.z_dim = z_dim

        self.linear1 = nn.Linear(input_size, h_dim)
        self.batchnorm1 = nn.BatchNorm1d(h_dim)
        self.merge = GaussianMerge(h_dim, z_dim)

        self.linear2 = nn.Linear(input_size, h_dim)
        self.batchnorm2 = nn.BatchNorm1d(h_dim)
        self.sample = GaussianSample(h_dim, z_dim)

    def forward(self, x, l_mu=None, l_log_var=None):
        if l_mu is not None:
            # Sample from this encode layer and merge
            z = self.linear1(x)
            z = F.relu(z, 0.1)
            z = self.batchnorm1(z)
            z = F.dropout(z)
            q_z, q_mu, q_log_var = self.merge(z, l_mu, l_log_var)

        # Sample from the decoder and send forward
        z = self.linear2(x)
        z = F.relu(z, 0.1)
        z = self.batchnorm2(z)
        z = F.dropout(z)
        z, p_mu, p_log_var = self.sample(z)

        if l_mu is None:
            return z

        return z, (q_z, (q_mu, q_log_var), (p_mu, p_log_var))

from models.generative.autoencoders.vae.sylvester_vae import SylvesterVAE
from models.generative.autoencoders.vae.vae import VariationalAutoencoder as VAE

class LadderVariationalAutoencoder(VAE):
    def __init__(self, flavour, z_dims, h_dims, auxiliary, a_dim=0, n_flows=0, input_size=None, early_stopping=250, warmup=10,
                 num_elements=None, hebb_layers=False):
        """
        Ladder Variational Autoencoder as described by
        [Sønderby 2016]. Adds several stochastic
        layers to improve the log-likelihood estimate.

        :param dims: x, z and hidden dimensions of the networks
        """
        super(LadderVariationalAutoencoder, self).__init__(flavour, z_dims, h_dims, auxiliary, n_flows=n_flows,
                                                           num_elements=num_elements, a_dim=a_dim)
        self.num_ortho_vecs = num_elements
        self.input_size = input_size
        self.early_stopping = early_stopping
        self.warmup = warmup
        self.auxiliary = auxiliary
        self.flow = None
        self.z_dims, self.h_dims = z_dims, h_dims
        self.encoder = None
        self.decoder = None
        self.kl_divergence = 0
        self.z_dim = self.z_dims[0]
        self.ladder = True
        if n_flows > 0 and not self.sylvester_flow:
            print("ADDING", flavour, "FLOWS", self.z_dims)
            self.add_flow(self.flow_flavour(in_features=self.z_dims, n_flows=n_flows, h_last_dim=self.h_dims[-1]))
        elif self.sylvester_flow:
            print("Adding Variational AutoEncoder Sylvester Flow")
            self.add_flow(self.flow_flavour(flow_flavour=flavour, in_features=self.z_dims, n_flows=n_flows,
                                            h_last_dim=h_dims[-1], auxiliary=auxiliary))
        else:
            self.n_flows = 0


    def set_lvae_layers(self):
        neurons = [self.input_size, *self.h_dims]
        encoder_layers = [LadderEncoder(neurons[i - 1], neurons[i], self.z_dims[i - 1]) for i in range(1, len(neurons))]
        decoder_layers = [LadderDecoder(self.z_dims[i - 1], self.h_dims[i - 1], self.z_dims[i]) for i in range(1, len(self.h_dims))][::-1]

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = Decoder(self.z_dims[0], self.h_dims, self.input_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, i=None):
        # Gather latent representation
        # from encoders along with final z.
        latents = []
        x = torch.tanh(x)
        if self.flow_type in ["o-sylvester", "h-sylvester", "t-sylvester"]:
            for i in range(len(self.encoder)):
                q_param, x, z = self.encoder(x, i)
                latents.append(q_param)
        else:
            for i, encoder in enumerate(self.encoder):
                q_param, x = encoder(x)
                z = q_param[0]
                q_param = q_param[1:]
                latents.append(q_param)
        latents = list(reversed(latents))
        kl_divergence = 0
        h = x
        self.log_det_j = 0
        for k, decoder in enumerate([-1, *self.decoder]):
            # If at top, encode == decoder,
            # use prior for KL.
            q_param = latents[k]
            if self.sylvester_flow:
                mu, log_var, r1, r2, q, b = q_param
                if k > 0:
                    z = [self.reparameterize(mu, log_var)]
                else:
                    z = [z]
                l = -1-k
                q_ortho = self.batch_construct_orthogonal(q, l)

                # Sample z_0
                # Normalizing flows
                for i in range(self.n_flows):
                    flow_k = getattr(self, 'flow_' + str(k) + "_" + str(i))
                    z_k, log_det_jacobian = flow_k(z[i], r1[:, :, :, i], r2[:, :, :, i], q_ortho[i, :, :, :],
                                                   b[:, :, :, i])

                    z.append(z_k)
                    self.log_det_j += log_det_jacobian

                # KL
                log_p_zk = log_standard_gaussian(z[-1])
                # ln q(z_0)  (not averaged)
                #mu, log_var, r1, r2, q, b = q_param_inverse
                log_q_z0 = log_gaussian(z[0], mu, log_var=log_var) - self.log_det_j
                # N E_q0[ ln q(z_0) - ln p(z_k) ]
                kl = log_q_z0 - log_p_zk
                kl_divergence += kl
                # x_mean = self.sample(z[-1])

            elif k == 0:
                kl_divergence += self._kld(z, q_param=q_param, i=k, h_last=h).abs()
            else:
                #q = (q_param_inverse[0], q_param_inverse[1])
                (mu, log_var) = q_param
                z, kl = decoder(z, mu, log_var)
                (q_z, q_param, p_param) = kl
                kl_divergence += self._kld(z, q_param=q_param, i=k, h_last=h, p_param=p_param).abs()
        try:
            x_mu = self.reconstruction(z)
        except:
            x_mu = self.reconstruction(z[-1])
        del latents, x, self.log_det_j, r1, r2, q, b, q_ortho, q_param
        self.kl_divergence = Variable(kl_divergence)
        return x_mu, z

    def sample(self, z):
        for decoder in self.decoder:
            z = decoder(z)
        return self.reconstruction(z)


class LadderSylvesterVariationalAutoencoder(SylvesterVAE):
    def __init__(self, flavour, z_dims, h_dims, auxiliary, a_dim=0, n_flows=0, input_size=None, early_stopping=250, warmup=10,
                 num_elements=None, hebb_layers=False):
        """
        Ladder Variational Autoencoder as described by
        [Sønderby 2016]. Adds several stochastic
        layers to improve the log-likelihood estimate.

        :param dims: x, z and hidden dimensions of the networks
        """
        super(LadderVariationalAutoencoder, self).__init__(flavour, z_dims, h_dims, auxiliary, n_flows=n_flows,
                                                           num_elements=num_elements, a_dim=a_dim)
        self.num_ortho_vecs = num_elements
        self.input_size = input_size
        self.early_stopping = early_stopping
        self.warmup = warmup
        self.auxiliary = auxiliary
        self.flow = None
        self.z_dims, self.h_dims = z_dims, h_dims
        self.encoder = None
        self.decoder = None
        self.kl_divergence = 0
        self.z_dim = self.z_dims[0]
        self.ladder = True
        if n_flows > 0 and not self.sylvester_flow:
            print("ADDING", flavour, "FLOWS", self.z_dims)
            self.add_flow(self.flow_flavour(in_features=self.z_dims, n_flows=n_flows, h_last_dim=self.h_dims[-1]))
        elif self.sylvester_flow:
            print("Adding Variational AutoEncoder Sylvester Flow")
            self.add_flow(self.flow_flavour(flow_flavour=flavour, in_features=self.z_dims, n_flows=n_flows,
                                            h_last_dim=h_dims[-1], auxiliary=auxiliary))
        else:
            self.n_flows = 0


    def set_lvae_layers(self):
        neurons = [self.input_size, *self.h_dims]
        encoder_layers = [LadderEncoder(neurons[i - 1], neurons[i], self.z_dims[i - 1]) for i in range(1, len(neurons))]
        decoder_layers = [LadderDecoder(self.z_dims[i - 1], self.h_dims[i - 1], self.z_dims[i]) for i in range(1, len(self.h_dims))][::-1]

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = Decoder(self.z_dims[0], self.h_dims, self.input_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, i=None):
        # Gather latent representation
        # from encoders along with final z.
        latents = []
        x = torch.tanh(x)
        if self.flow_type in ["o-sylvester", "h-sylvester", "t-sylvester"]:
            for i in range(len(self.encoder)):
                q_param, x, z = self.encoder(x, i)
                latents.append(q_param)
        else:
            for i, encoder in enumerate(self.encoder):
                q_param, x = encoder(x)
                z = q_param[0]
                q_param = q_param[1:]
                latents.append(q_param)
        latents = list(reversed(latents))
        kl_divergence = 0
        h = x
        self.log_det_j = 0
        for k, decoder in enumerate([-1, *self.decoder]):
            # If at top, encode == decoder,
            # use prior for KL.
            q_param = latents[k]
            if self.sylvester_flow:
                mu, log_var, r1, r2, q, b = q_param
                if k > 0:
                    z = [self.reparameterize(mu, log_var)]
                else:
                    z = [z]
                l = -1-k
                q_ortho = self.batch_construct_orthogonal(q, l)

                # Sample z_0
                # Normalizing flows
                for i in range(self.n_flows):
                    flow_k = getattr(self, 'flow_' + str(k) + "_" + str(i))
                    z_k, log_det_jacobian = flow_k(z[i], r1[:, :, :, i], r2[:, :, :, i], q_ortho[i, :, :, :],
                                                   b[:, :, :, i])

                    z.append(z_k)
                    self.log_det_j += log_det_jacobian

                # KL
                log_p_zk = log_standard_gaussian(z[-1])
                # ln q(z_0)  (not averaged)
                #mu, log_var, r1, r2, q, b = q_param_inverse
                log_q_z0 = log_gaussian(z[0], mu, log_var=log_var) - self.log_det_j
                # N E_q0[ ln q(z_0) - ln p(z_k) ]
                kl = log_q_z0 - log_p_zk
                kl_divergence += kl
                # x_mean = self.sample(z[-1])

            elif k == 0:
                kl_divergence += self._kld(z, q_param=q_param, i=k, h_last=h).abs()
            else:
                #q = (q_param_inverse[0], q_param_inverse[1])
                (mu, log_var) = q_param
                z, kl = decoder(z, mu, log_var)
                (q_z, q_param, p_param) = kl
                kl_divergence += self._kld(z, q_param=q_param, i=k, h_last=h, p_param=p_param).abs()
        try:
            x_mu = self.reconstruction(z)
        except:
            x_mu = self.reconstruction(z[-1])
        del latents, x, self.log_det_j, r1, r2, q, b, q_ortho, q_param
        self.kl_divergence = Variable(kl_divergence)
        return x_mu, z

    def sample(self, z):
        for decoder in self.decoder:
            z = decoder(z)
        return self.reconstruction(z)
