from __future__ import print_function
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
import torch
import keras
import numpy as np
from torch import optim
#from utils.utils import
if(torch.cuda.is_available()):
    cuda=True
else:
    cuda=False
from utils.losses import FreeEnergyBound
from utils.flow import NormalizingFlow
from utils.densities import p_z
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dimension_reduction.pca import pca2d, pca_labels

import math
from torch.autograd import Variable
import torch.nn.functional as F

n_epochs = 100
batch_size = 64
nrep = 1
lr = 0.015
mom=0.5
latent_size = 3
nflows = 32
normflow = False
householder_flow = False
scheduler = False
checkpoint=False
np.random.seed(1111)
torch.manual_seed(42)
epoch_interval = 5
Labels=True
hflow = True


# the data, shuffled and split between train and test sets
data1 = np.load('data/dementia_gene_expression.pickle.npy')
data1[np.isnan(data1)] = 0
labels = np.load('data/dementia_labels.pickle.npy')

dict_labels = dict(zip(list(range(len(labels))),labels))
num_classes = 3


def elbo(recon_x, x, mu, logvar):
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    BCE = F.binary_cross_entropy(recon_x.view(-1), x.view(-1), size_average=False)
    KLD = torch.sum(0.5 * (mu.data ** 2 + torch.exp(logvar.data) - 1 - logvar.data))
    del recon_x, x, mu, logvar
    return BCE + KLD

def normflowELBO(recon_x, x, KLD):
    assert KLD is not None
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    del recon_x, x
    return BCE + KLD


def reparameterize( mu, logvar, training = True):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu


class VAE(nn.Module):
    def __init__(self, n_inputs, latent_size=2):
        super(VAE, self).__init__()
        # will store values to plot; 2d representation
        self.z = []
        self.latent_size = latent_size
        # Module lists
        self.reduce1 = nn.Linear(n_inputs, 1024)
        self.reduce1.weight = nn.init.kaiming_uniform_(self.reduce1.weight)

        self.reduce1_bn = nn.BatchNorm1d(1024)
        self.reduce2 = nn.Linear(1024, 128)
        self.reduce2.weight = nn.init.kaiming_uniform_(self.reduce2.weight)
        self.reduce2_bn = nn.BatchNorm1d(128)
        self.reduce_neck_mu = nn.Linear(128, self.latent_size)
        self.reduce_neck_var = nn.Linear(128, self.latent_size)
        self.reduce_neck_bn = nn.BatchNorm1d(self.latent_size)
        self.reduce_neck_mu.weight = nn.init.kaiming_uniform_(self.reduce_neck_mu.weight)
        self.reduce_neck_var.weight = nn.init.kaiming_uniform_(self.reduce_neck_var.weight)
        self.reduce_neck_mu_bn = nn.BatchNorm1d(self.latent_size)
        self.reduce_neck_var_bn = nn.BatchNorm1d(self.latent_size)
        self.upscale_neck = nn.Linear(self.latent_size, 128)
        self.upscale_neck.weight = nn.init.kaiming_uniform_(self.upscale_neck.weight)
        self.upscale_neck_bn = nn.BatchNorm1d(128)
        self.upscale1 = nn.Linear(128, 1024)
        self.upscale1.weight = nn.init.kaiming_uniform_(self.upscale1.weight)
        self.upscale1_bn = nn.BatchNorm1d(1024)
        self.upscale2 = nn.Linear(1024, n_inputs)
        self.upscale2.weight = nn.init.kaiming_uniform_(self.upscale2.weight)
        self.upscale2_bn = nn.BatchNorm1d(n_inputs)

        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def plot_z(self, allLabels,lr,mom,batch_size,nf=True):
        zs = np.vstack(model.z)
        labs = np.vstack(allLabels)
        fig, ax = plt.subplots()  # create figure and axis
        ax = fig.add_subplot(111, projection='3d')
        z001 = []
        z002 = []
        z003 = []
        z010 = []
        z020 = []
        z030 = []
        z100 = []
        z200 = []
        z300 = []
        colors = []
        for Z, label in zip(zs, labs):
            if (label[0] == 1):
                if (Z[0] >= 0):
                    z001 += [np.log(1 + Z[0])]
                else:
                    z001 += [-np.log(1 + np.absolute(Z[0]))]
                if (Z[1] >= 0):
                    z010 += [np.log(1 + Z[1])]
                else:
                    z010 += [-np.log(1 + np.absolute(Z[1]))]
                if (Z[2] >= 0):
                    z100 += [np.log(1 + Z[2])]
                else:
                    z100 += [-np.log(1 + np.absolute(Z[2]))]

            elif (label[1] == 1):
                if (Z[0] >= 0):
                    z002 += [np.log(1 + Z[0])]
                else:
                    z002 += [-np.log(1 + np.absolute(Z[0]))]
                if (Z[1] >= 0):
                    z020 += [np.log(1 + Z[1])]
                else:
                    z020 += [-np.log(1 + np.absolute(Z[1]))]
                if (Z[2] >= 0):
                    z200 += [np.log(1 + Z[2])]
                else:
                    z200 += [-np.log(1 + np.absolute(Z[2]))]
            elif (label[2] == 1):
                if (Z[0] >= 0):
                    z003 += [np.log(1 + Z[0])]
                else:
                    z003 += [-np.log(1 + np.absolute(Z[0]))]
                if (Z[1] >= 0):
                    z030 += [np.log(1 + Z[1])]
                else:
                    z030 += [-np.log(1 + np.absolute(Z[1]))]
                if (Z[2] >= 0):
                    z300 += [np.log(1 + Z[2])]
                else:
                    z300 += [-np.log(1 + np.absolute(Z[2]))]
        # labels in label_set_list should be opposite, don'T undeerstand why inversed, like this it works, but should be modified
        ax.scatter(z001, z010,z100, '.', c="r", label=label_set_list[0],s=3)
        ax.scatter(z002, z020,z200, '.', c="g", label=label_set_list[1],s=3)
        ax.scatter(z003, z030,z300, '.', c="b", label=label_set_list[2],s=3)

        #Axes3D.scatter(z03, z13, zs=0, zdir='z', s=20, c=None, depthshade=True)¶

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        plt.tight_layout()
        fig.tight_layout()
        fig.savefig("VAE" + str(epoch) + filenames[0] + '_lr'+str(lr)+ '_mom'+str(mom)+ '_bs'+str(batch_size)+ '_nf'+str(nf) +".png")
        plt.close(fig)

    def reconstruct_from_z(self, z):

        x = self.upscale_neck(z)
        x = self.upscale_neck_bn(x)
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.upscale1(x)
        x = self.upscale1_bn(x)
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.upscale2(x)
        x = self.upscale2_bn(x)
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.sigmoid(x)


        return x

    def forward(self, x):
        x = self.reduce1(x)
        x = self.reduce1_bn(x)
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.reduce2(x)
        x = self.reduce2_bn(x)
        x = self.lrelu(x)
        x = self.dropout(x)

        #mu = self.reduce_neck_mu_bn(self.reduce_neck_mu(x))
        #mu = self.dropout(mu)
        #var = self.reduce_neck_var_bn(self.reduce_neck_var(x))
        #var = self.dropout(var)

        mu = self.reduce_neck_mu(x)
        var = self.reduce_neck_var(x)

        z = reparameterize(mu, var)
        z = self.reduce_neck_bn(z)
        self.z.append(z.data.cpu().numpy())

        z = self.lrelu(z)
        z = self.dropout(z)

        x = self.upscale_neck(z)
        x = self.upscale_neck_bn(x)
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.upscale1(x)
        x = self.upscale1_bn(x)
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.upscale2(x)
        x = self.upscale2_bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.sigmoid(x)


        return x, z, mu,var


# the data, shuffled and split between train and test sets
data1 = np.load('data/dementia_gene_expression.pickle.npy')
data1[np.isnan(data1)] = 0
labels = np.load('data/dementia_labels.pickle.npy')

dict_labels = dict(zip(list(range(len(labels))),labels))
num_classes = 3

# Replace labels with 0,1,2 for not demented, alzheimer and hungtington's disease, respectively
filenames = ['controlHuntingtonAlzheimer_history']
Labels = [labels]
accuracy_training_array = []
accuracy_valid_array = []
losses_training_array = []
losses_valid_array = []
max_valid_accuracies = []
max_valid_epochs = []
min_valid_loss = []
min_valid_loss_epochs = []
labels = Labels
labelset = []
label_set = set(Labels[0])

classes = np.copy(labels[0])
label_set_list = []
for index, label in enumerate(label_set):
    print(label)
    if(label not in label_set_list):
        label_set_list += [label]
    for lab in range(len(labels[0])):
        if (label == labels[0][lab]):
            classes[lab] = int(index)
label_set_num = dict(zip(range(len(label_set)),label_set))
# Labels are in order of appearance in label_set_list
print('label list:',label_set_list)
classes = np.array(classes)
random_training = np.random.choice(range(len(classes)),size=(int(len(classes)*(5/6))),replace=False)
x_train = data1[:,random_training]
x_test = np.delete(data1,random_training,axis=1)
y_train_1 = y_train = classes[random_training]
y_test = np.array(np.delete(classes,random_training))


x_train = np.transpose(x_train)
x_test = np.transpose(x_test)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# this is the size of our encoded representations

targets = pca_labels(y_train)
pca2d(x_train,y=targets)

train_set = torch.utils.data.TensorDataset(torch.FloatTensor(x_train),torch.FloatTensor(y_train))
test_set = torch.utils.data.TensorDataset(torch.FloatTensor(x_test),torch.FloatTensor(y_test))
train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size)
model = VAE(data1.shape[0],latent_size=latent_size)
if cuda:
    model.cuda()

if normflow:
    print("Norm Flow",normflow)
    flow = NormalizingFlow(dim=latent_size, flow_length=nflows)
    optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=mom)
    bound = FreeEnergyBound(density=p_z)

elif hflow:
    print("Householder Flow", normflow)
    optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=mom)
else:
    print("Vanilla")
    optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=mom)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
for epoch in range(n_epochs):
    model.z = []
    allLabels = []
    train_loss = 0
    count = 0
    for batch_idx, (data, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        count = count + 1

        if Labels is True:
            allLabels.append(labels)
        data = Variable(data)
        if cuda:
            data = data.cuda()
        data = F.sigmoid(data)
        result, z, mu, logvar = model(data)
        if hflow:
            z = {}
            # z ~ q(z | x)
            z_q_mean, z_q_logvar, h_last = q_z(model,data)
            z['0'] = model.reparameterize(z_q_mean, z_q_logvar)
            # Householder Flow:
            z = hq_z_Flow(z, h_last)

            # x_mean = p(x|z)
            mu, logvar = p_x(z['1'])

            loss = elbo(result, data, mu, logvar)

            del mu, logvar, z_q_mean, z_q_logvar, h_last
        elif normflow:
            zk, log_jacobians = flow(z)
            KLD = bound(zk, log_jacobians)
            recon_batch_zk = model.reconstruct_from_z(zk)
            loss = normflowELBO(recon_batch_zk, data, KLD.clone())
            del zk, log_jacobians, KLD, recon_batch_zk
        else:
            loss = elbo(result, data, mu, logvar)

        try:
            assert np.isnan(loss.data.item()) == False
        except:
            print(loss.data.item())
        scheduler.step(loss.data.item())
        loss.backward()
        clip_grads(model)
        train_loss += loss.item()
        optimizer.step()
    if epoch % epoch_interval == 0:
        print('====> Epoch: {} Average loss: {:.12f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        '''
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

        '''
        model.plot_z(allLabels, lr,mom, batch_size=batch_size,nf=normflow)
    tmp = []
    zs = []
    plot_z = True
