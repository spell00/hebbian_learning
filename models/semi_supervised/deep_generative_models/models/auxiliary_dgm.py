import torch
import torch.nn as nn
import torch.nn.init as init
from models.generative.autoencoders.vae.vae import Encoder, Decoder
from models.semi_supervised.deep_generative_models.models.dgm import DeepGenerativeModel, MLP, ConvNet, ConvDecoder, ConvEncoder
import numpy as np

ladder = False #TODO for some reason didnt load correctly from list_parameters
if ladder:
    from models.semi_supervised.deep_generative_models.models import LadderDeepGenerativeModel as DeepGenerativeModel


class AuxiliaryDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, flow_type, z_dims, h_dims, n_flows, a_dim, num_elements, dropout=0.5,
                 is_hebb_layers=False, gt_input=-100, use_conv=False):
        """
        Auxiliary Deep Generative Models [Maaløe 2016]
        code replication. The ADGM introduces an additional
        latent variable 'a', which enables the model to fit
        more complex variational distributions.

        :param dims: dimensions of x, y, z, a and hidden layers.
        """
        if use_conv:
            use_conv = False
            print("NOT USING CONVOLUTIONS; NOT PROGRAMMED!")
        self.gt_input = gt_input
        self.auxiliary = True
        self.hebb_layers = is_hebb_layers
        self.z_dim = z_dims[-1]
        super(AuxiliaryDeepGenerativeModel, self).__init__(flow_type=flow_type, z_dims=z_dims, h_dims=h_dims,
                                                           n_flows=n_flows, a_dim=a_dim, auxiliary=self.auxiliary,
                                                           num_elements=num_elements, dropout=dropout,
                                                           hebb_layers=is_hebb_layers, gt_input=gt_input)
        if type(z_dims) is not list:
            self.z_dim = z_dims
            self.z_dims = [z_dims]
        else:
            self.z_dims = z_dims = z_dims
            self.z_dim = z_dims[-1]
        if not self.ladder:
            self.z_dims = [self.z_dims[-1]]
        self.a_dim = a_dim

    def classify(self, x, valid_bool, input_pruning=True, start_pruning=-1):
        # Auxiliary inference q(a|x)
        (a, a_mu, a_log_var), _ = self.aux_encoder(x, input_shape=self.input_shape)

        # Classification q(y|a,x)
        if input_pruning is True and self.epoch >= start_pruning and start_pruning > -1:
            valid_bool = torch.Tensor(valid_bool).cuda()
        logits = self.classifier(x, a=a, valid_bool=valid_bool)
        return logits

    def set_adgm_layers(self, h_dims, input_shape, is_hebb_layers=False, use_conv_classifier=False,
                        planes_classifier=None, classifier_kernels=None, classifier_pooling_layers=None):
        if use_conv_classifier:
            self.set_dgm_layers(input_shape=input_shape)
            self.classifier = ConvNet(input_shape=self.input_shape, h_dims=h_dims, num_classes=self.num_classes,
                                      planes=planes_classifier, kernels=classifier_kernels,
                                      pooling_layers=classifier_pooling_layers, a_dim=self.a_dim)
        else:
            self.set_dgm_layers(input_shape=input_shape, num_classes=self.num_classes, is_hebb_layers=is_hebb_layers)
            self.classifier = MLP(self.input_size, self.input_shape, self.indices_names, h_dims, self.num_classes,
                                  a_dim=self.a_dim, is_hebb_layers=is_hebb_layers, gt_input=self.gt_input,
                                  extra_class=False)

        self.aux_encoder = Encoder(self.input_size, self.h_dims, self.a_dim, num_classes=self.num_classes, y_dim=0)

        self.aux_decoder = Encoder(self.input_size + self.z_dim, list(reversed(self.h_dims)),
                                   self.a_dim, num_classes=self.num_classes, y_dim=self.num_classes)

        self.encoder = Encoder(input_size=self.input_size, h_dim=self.h_dims, z_dim=self.z_dim,
                               num_classes=self.num_classes, a_dim=self.a_dim, y_dim=self.num_classes)
        self.decoder = Decoder(self.z_dim, list(reversed(self.h_dims)), self.input_size, num_classes=self.num_classes)
        self.add_flow_auxiliary(self.flow_flavour(in_features=[self.a_dim], n_flows=self.n_flows, h_last_dim=h_dims[-1],
                                auxiliary=True, flow_flavour=self.flavour))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def set_conv_adgm_layers(self, hs_ae, h_dims, planes_ae, kernels_ae, padding_ae,
                             pooling_layers_ae, planes_classifier=None, classifier_kernels=None,
                             classifier_pooling_layers=None, use_conv_classifier=True, input_shape=None,
                             is_hebb_layers=False):
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        if use_conv_classifier:
            self.set_conv_dgm_layers(hs_ae=hs_ae, hs_class=h_dims, z_dim=self.z_dim, planes_ae=planes_ae, kernels_ae=kernels_ae,
                                     padding_ae=padding_ae, pooling_layers_ae=pooling_layers_ae,
                                     planes_c=planes_classifier, kernels_c=classifier_kernels,
                                     pooling_layers_c=classifier_pooling_layers)
            self.classifier = ConvNet(input_shape=self.input_shape, h_dims=h_dims, num_classes=self.num_classes,
                                      planes=planes_classifier, kernels=classifier_kernels,
                                      pooling_layers=classifier_pooling_layers, a_dim=self.a_dim)
        else:
            self.set_dgm_layers(input_shape=self.input_shape, is_hebb_layers=is_hebb_layers)
            self.classifier = MLP(self.input_size, self.input_shape, h_dims, self.num_classes, is_hebb_layers=is_hebb_layers,
                                  a_dim=self.a_dim, gt=self.gt_input, num_classes=self.num_classes)

        self.aux_encoder = ConvEncoder(h_dim=hs_ae, z_dim=self.a_dim, planes=planes_ae, kernels=kernels_ae,
                                       padding=padding_ae, pooling_layers=pooling_layers_ae, y_size=0, a_size=0)
        self.aux_decoder = ConvEncoder(h_dim=list(reversed(hs_ae)), z_dim=self.a_dim, planes=planes_ae,
                                       kernels=kernels_ae, padding=padding_ae, pooling_layers=pooling_layers_ae,
                                       y_size=self.num_classes, a_size=self.z_dim)
        # self.aux_encoder = Encoder(input_size=self.input_size, h_dim=self.h_dims, z_dim=self.a_dim)
        # self.aux_decoder = Encoder(input_size=self.input_size + self.z_dim + self.num_classes,
        #                           h_dim=list(reversed(self.h_dims)), z_dim=self.a_dim)

        self.encoder = ConvEncoder(hs_ae, z_dim=self.z_dim, planes=planes_ae, kernels=kernels_ae,
                                   padding=padding_ae, pooling_layers=pooling_layers_ae, y_size=self.num_classes,
                                   a_size=self.a_dim)
        self.decoder = ConvDecoder(z_dim=self.z_dim, y_dim=self.num_classes, h_dim=list(reversed(hs_ae)),
                                   input_shape=self.input_shape, planes=planes_ae, kernels=kernels_ae,
                                   padding=padding_ae, unpooling_layers=list(reversed(pooling_layers_ae)))
        self.add_flow_auxiliary(self.flow_flavour(in_features=[self.a_dim], n_flows=self.n_flows, h_last_dim=h_dims[-1],
                                auxiliary=True, flow_flavour=self.flavour))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        """
        Forward through the model
        :param x: features
        :param y: labels
        :return: reconstruction
        """
        # Auxiliary inference q(a|x)
        x = torch.tanh(x)
        (q_a, q_a_mu, q_a_log_var), h_a0 = self.aux_encoder(x, input_shape=self.input_shape)
        # Latent inference q(z|a,y,x)
        (z, z_mu, z_log_var), h = self.encoder(x, input_shape=self.input_shape, y=y, a=q_a)

        # Generative p(x|z,y)
        x_mean = self.decoder(z, y)

        # Generative p(a|z,y,x)
        (p_a, p_a_mu, p_a_log_var), h_a = self.aux_decoder(x=x, input_shape=self.input_shape, y=y, a=z)

        a_kl = self._kld(q_a, q_param=(q_a_mu, q_a_log_var), i=0, h_last=h_a, p_param=(p_a_mu, p_a_log_var), auxiliary=True)
        z_kl = self._kld(z, q_param=(z_mu, z_log_var), i=0, h_last=h, auxiliary=False)

        self.kl_divergence = a_kl + z_kl
        del a_kl, z_kl, p_a, p_a_mu, p_a_log_var, h_a, x, z, z_mu, z_log_var, q_a, q_a_mu, q_a_log_var, h_a0, h
        return x_mean


    def run_sylvester_dgm(self, x, auxiliary, k=0, y=torch.Tensor([]).cuda()):
        """
        Forward pass with orthogonal sylvester flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """
        a = {0: None, 1: None}
        z = {0: None, 1: None}
        self.log_det_j = 0.
        x = torch.tanh(x)

        if auxiliary:
            _, q_a_mu, q_a_log_var, self.log_det_j, a[0], a[-1] = self.run_sylvester(x=x, k=k, auxiliary=True)
            a_kl = self.kl_divergence

            # VOIR SVI
            x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1] = self.run_sylvester(x=x, y=y, a=a[-1], k=k, auxiliary=False)
            z_kl = self.kl_divergence

            self.kl_divergence = a_kl + z_kl
            del a_kl, z_kl, q_a_mu, q_a_log_var, a
        else:
            x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1] = self.run_sylvester(x=x, y=y, k=k, auxiliary=False)
        del z, z_var, z_mu, x, y, k
        return x_mean





