# TODO: THIS IS NOT USED, BUT SHOULD BE EASY TO INCLUDE IN THE PIPELINE. IMPORTANT, THIS IS OFTEN USED AS A BENCHMARK TO
# TODO: COMPARE MORE SOPHISTICATED


import torch.nn as nn
from models.semi_supervised.deep_generative_models.layers.flow import IAF
from models.generative.autoencoders.vae.vae import VariationalAutoencoder


class IAFVAE(VariationalAutoencoder):
    """
    Variational auto-encode with inverse autoregressive flows in the encode.
    """

    def __init__(self):
        super(IAFVAE, self).__init__()

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.
        self.h_size = self.h_dims

        self.h_context = nn.Linear(self.h_dim[-1], self.h_size)

        # Flow parameters
        self.n_flows = self.n_flows
        self.flow = IAF(z_size=self.z_dim_last, n_flows=self.n_flows, num_hidden=1, h_size=self.h_size, conv2d=False)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and context h for flows.
        """

        h = self.encoder(x)
        h = h.view(-1, self.h_dim[-1])
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)
        h_context = self.h_context(h)

        return mean_z, var_z, h_context

    def forward(self, x):
        """
        Forward pass with inverse autoregressive flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        # mean and variance of z
        z_mu, z_var, h_context = self.encode(x)
        # sample z
        z_0 = self.reparameterize(z_mu, z_var)

        # iaf flows
        z_k, self.log_det_j = self.flow(z_0, h_context)

        # decode
        x_mean = self.decode(z_k)

        return x_mean, z_mu, z_var, self.log_det_j, z_0, z_k



