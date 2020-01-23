import torch.nn as nn
from torch.nn import init
import torch
from models.generative.autoencoders.vae.vae import Decoder
from models.generative.autoencoders.vae.ladder_vae import LadderEncoder, LadderDecoder
from models.semi_supervised.deep_generative_models.models.dgm import DeepGenerativeModel, MLP


class LadderDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, flow_type, z_dims, h_dims, n_flows, auxiliary, a_dim=None, num_elements=None, gt_input=-100):
        """
        Ladder version of the Deep Generative Model.
        Uses a hierarchical representation that is
        trained end-to-end to give very nice disentangled
        representations.

        :param dims: dimensions of x, y, z layers and h layers
            note that len(z) == len(h).
        """
        super(LadderDeepGenerativeModel, self).__init__(flow_type, z_dims, h_dims, n_flows, a_dim, auxiliary, num_elements=num_elements)
        try:
            assert len(z_dims) == len(h_dims)
        except:
            print("In a ladder VAE, the number of latent sizes should be the same as the number of hidden layers")
            print("The sizes received: h_dim ", len(h_dims), "z_dim_last:", len(z_dims))
            exit()
        self.auxiliary = auxiliary
        self.gt_input = gt_input
        self.z_dims, self.h_dim, self.n_flows = z_dims, h_dims, n_flows
        self.n_flows = n_flows
        self.a_dim = a_dim
        self.flow_type = flow_type

    def set_ldgm_layers(self):
        self.z_dim = self.z_dims[-1]
        neurons = [self.input_size, *self.h_dim]
        encoder_layers = [LadderEncoder(neurons[i - 1], neurons[i], self.z_dims[i - 1]) for i in range(1, len(neurons))]

        e = encoder_layers[-1]
        encoder_layers[-1] = LadderEncoder(e.in_features + self.num_classes, e.out_features, e.z_dim)

        decoder_layers = [LadderDecoder(self.z_dims[i - 1], self.h_dim[i - 1], self.z_dims[i]) for i in range(1, len(self.h_dim))][::-1]

        h_dims = [self.h_dim[0] for _ in range(1)]

        self.classifier = MLP(self.input_size, h_dims, self.num_classes)

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = Decoder(self.z_dims[0]+self.num_classes, self.h_dim, self.input_size)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # Gather latent representation
        # from encoders along with final z.
        x = torch.tanh(x)
        latents = []
        for i, encoder in enumerate(self.encoder):
            if i == len(self.encoder)-1:
                (z, mu, log_var), x = encoder(torch.cat([x, y], dim=1))
            else:
                (z, mu, log_var), x = encoder(x)
            latents.append((mu, log_var))

        latents = list(reversed(latents))

        self.kl_divergence = 0
        for i, decoder in enumerate([-1, *self.decoder]):
            # If at top, encode == decoder,
            # use prior for KL.
            l_mu, l_log_var = latents[i]
            if i == 0:
                self.kl_divergence += self._kld(z, (l_mu, l_log_var), i=i, h_last=x)

            # Perform downword merge of information.
            else:
                z, kl = decoder(z, l_mu, l_log_var)
                (q_z, q_param, p_param) = kl
                self.kl_divergence += self._kld(z, q_param=q_param, i=i, h_last=x, p_param=p_param).abs()
        x_mu = self.reconstruction(torch.cat([z, y], dim=1))
        return x_mu

    def sample(self, z, y):
        for i, decoder in enumerate(self.decoder):
            z = decoder(z)
        return self.reconstruction(torch.cat([z, y], dim=1))