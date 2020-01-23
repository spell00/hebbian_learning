import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

def hebb_values_transform(tensor, on_zero):
    # we create a copy of the original tensor,
    # because of the way we are replacing them.
    res = tensor.clone()
    res.data[res.data == 0] = on_zero
    return res

def hebb_values_transform_conv(tensor, on_zero):
    # we create a copy of the original tensor,
    # because of the way we are replacing them.
    res = tensor.clone()
    res.data[res.data == 0] = on_zero

    res = torch.sum(res,dim=2)
    res = torch.sum(res,dim=2)
    return res

def hebb_values_negative_transform(tensor,multiplier=0.1):
    # we create a copy of the original tensor,
    # because of the way we are replacing them.
    res = tensor.clone()
    res.data[res.data < 0] = multiplier*res.data[res.data < 0]
    return res

def indices(array):
    indices = []
    for ind,x in enumerate(array):
        if(int(x) == 1):
            indices.append(ind)
    return indices

def indices_conv(array):
    indices = []
    for ind,x in enumerate(array):
        if(x == 1):
            indices.append(ind)
    return indices


def hebbian_sort_pruning_values(net,nremove):
    GTs = [[]] * len(nremove)
    for i in range(len(GTs)):
        hebb = Variable(net.hebbs[i].data.copy_(net.hebbs[i].data))
        sorted_hebb = np.sort(hebb.data)
        GTs[i] = sorted_hebb[nremove[i]]
    return GTs


def hebbian_pruning(net, GT, min_nclasses = 2, minimum_neurons=2):
    hebb = Variable(net.hebbs[0].data.copy_(net.hebbs[0].data))
    if(len(hebb) >= min_nclasses):
        toKeep = hebb > float(GT[0])
        toKeep_array = toKeep.data == 1
        toKeep_indices = indices(toKeep_array)
        if (len(toKeep_indices) < minimum_neurons):
            # TODO Replace neurons that could not be removed?
            toKeep_indices = indices(torch.sort(hebb)[1] < minimum_neurons)
            print("Minimum neurons on layer 1")
        net.hebbs[0] = hebb[toKeep_indices]

        w1 = net.fc1.weight.data.copy_(net.fc1.weight.data).cpu().numpy()
        b1 = net.fc1.bias.data.copy_(net.fc1.bias.data).cpu().numpy()
        wg1 = net.fc1.weight.grad.data.copy_(net.fc1.weight.grad.data).cpu().numpy()
        wb1 = net.fc1.bias.grad.data.copy_(net.fc1.bias.grad.data).cpu().numpy()

        w1 = w1[toKeep_indices, :]
        b1 = b1[toKeep_indices]

        wg1 = wg1[toKeep_indices, :]
        wb1 = wb1[toKeep_indices]
        wg1 = torch.from_numpy(wg1)
        wb1 = torch.from_numpy(wb1)
        b1 = torch.from_numpy(b1)
        w1 = torch.from_numpy(w1)
        if (torch.cuda.is_available()):
            wg1 = wg1.cuda()
            wb1 = wb1.cuda()
            b1 = b1.cuda()
            w1 = w1.cuda()
        net.fc1.weight.data = torch.Tensor(w1)
        net.fc1.bias.data = torch.Tensor(b1)
        net.fc1.in_features = len(w1[0])
        net.fc1.out_features = len(w1)
        net.fc1.weight.grad = torch.nn.Parameter(wg1)
        net.fc1.bias.grad = torch.nn.Parameter(wb1)
        net.bns[0] = nn.BatchNorm1d(len(net.fc1.bias))

    for i in range(len(GT)-1):
        hebb2 = Variable(net.hebbs[i+1].data.copy_(net.hebbs[i+1].data))
        if (len(hebb2) >= min_nclasses):
            toKeep2 = hebb2 > float(GT[i+1])
            toKeep2_array = toKeep2.data == 1
            toKeep2_indices = indices(toKeep2_array)
            if (len(toKeep2_indices) < minimum_neurons):
                # TODO Replace neurons that could not be removed?
                toKeep2_indices = indices(torch.sort(hebb2)[1] < minimum_neurons)
                print("Minimum neurons on layer ", (i + 2))

            net.hebbs[i+1] = hebb2[toKeep2_indices]
            w2 = net.fcs[i].weight.data.copy_(net.fcs[i].weight.data).cpu().numpy()
            b2 = net.fcs[i].bias.data.copy_(net.fcs[i].bias.data).cpu().numpy()

            wg2 = net.fcs[i].weight.grad.data.copy_(net.fcs[i].weight.grad.data).cpu().numpy()
            wb2 = net.fcs[i].bias.grad.data.copy_(net.fcs[i].bias.grad.data).cpu().numpy()
            wg2 = wg2[toKeep2_indices,:]
            wg2 = wg2[:,toKeep_indices]
            wb2 = wb2[toKeep2_indices]

            wg2 = torch.from_numpy(wg2)
            wb2 = torch.from_numpy(wb2)


            w2 = w2[toKeep2_indices,:]
            w2 = w2[:,toKeep_indices]
            w2 = torch.from_numpy(w2)
            b2 = b2[toKeep2_indices]
            b2 = torch.from_numpy(b2)
            if (torch.cuda.is_available()):
                wg2 = wg2.cuda()
                w2 = w2.cuda()
                b2 = b2.cuda()
                wb2 = wb2.cuda()
            net.fcs[i].weight.data = torch.Tensor(w2)
            net.fcs[i].bias.data = torch.Tensor(b2)
            net.fcs[i].in_features = len(w2[0])
            net.fcs[i].out_features = len(w2)
            net.fcs[i].weight.grad = torch.nn.Parameter(wg2)
            net.fcs[i].bias.grad = torch.nn.Parameter(wb2)
            net.bns[i+1] = nn.BatchNorm1d(len(net.fcs[i].bias))
            toKeep_indices = toKeep2_indices
    if (len(hebb) > min_nclasses):
        w3 = net.fclast.weight.data.copy_(net.fclast.weight.data).cpu().numpy()
        w3 = torch.from_numpy(w3[:,toKeep2_indices])

        gw3 = net.fclast.weight.grad.data.copy_(net.fclast.weight.grad.data).cpu().numpy()
        gw3 = gw3[:,toKeep2_indices]
        gw3 = torch.from_numpy(gw3)
        if (torch.cuda.is_available()):
            w3 = w3.cuda()
            gw3 = gw3.cuda()
        net.fclast.weight = torch.nn.Parameter(w3)
        net.fclast.in_features = len(w3[0])
        net.fclast.weight.grad = torch.nn.Parameter(gw3)
    if(torch.cuda.is_available()):
        net = net.cuda()
    return net

def hebbian_pruning_conv(net, GT_convs=[0,0,0,0,0,0] ,  min_neurons=4  ):
    hebb_conv = net.hebbs_conv[0].data.copy_(net.hebbs_conv[0].data)
    toKeep = hebb_conv > float(GT_convs[0])
    toKeep_array = toKeep == 1
    toKeep_indices = indices_conv(toKeep_array)
    if(len(toKeep_indices) < min_neurons):
        # TODO Replace neurons that could not be removed?
        print("Minimum neurons on layer 1")
        toKeep_indices = indices_conv(torch.sort(hebb_conv)[1] < min_neurons)
    net.hebbs_conv[0] = Variable(hebb_conv[toKeep_indices])

    w1 = net.convs[0].weight
    b1 = net.convs[0].bias
    weight1 = w1.data[toKeep_indices, :]
    bias1 = b1.data[toKeep_indices]
    gw1 = net.convs[0].weight.grad[toKeep_indices, :]
    gb1 = net.convs[0].bias.grad[toKeep_indices]

    net.convs[0].weight = torch.nn.Parameter(weight1)
    net.convs[0].bias = torch.nn.Parameter(bias1)
    net.convs[0].in_channels = len(weight1[0])
    net.convs[0].out_channels = len(weight1)
    net.convs[0].weight.grad = gw1
    net.convs[0].bias.grad = gb1

    net.bns[0] = nn.BatchNorm1d(len(net.convs[0].bias))

    for i in range(1,len(GT_convs)):
        hebb2 = net.hebbs_conv[i].data.copy_(net.hebbs_conv[i].data)
        toKeep2 = hebb2 > float(GT_convs[i])
        toKeep2_array = toKeep2 == 1
        toKeep2_indices = indices_conv(toKeep2_array)
        if (len(toKeep2_indices) < min_neurons):
            # TODO Replace neurons that could not be removed?
            toKeep2_indices = indices_conv(torch.sort(hebb2)[1] < min_neurons)
            print("Minimum neurons on layer ",(i+1))

        net.hebbs_conv[i] = Variable(hebb2[toKeep2_indices])
        w2 = net.convs[i].weight.data.copy_(net.convs[i].weight.data).cpu().numpy()
        b2 = net.convs[i].bias.data.copy_(net.convs[i].bias.data).cpu().numpy()

        gw2 = net.convs[i].weight.grad.data.copy_(net.convs[i].weight.grad.data).cpu().numpy()
        gb2 = net.convs[i].bias.data.copy_(net.convs[i].bias.grad.data).cpu().numpy()
        gb2 = gb2[toKeep2_indices]

        gw2 = gw2[toKeep2_indices,:]
        gw2 = gw2[:,toKeep_indices]
        gw2 = torch.from_numpy(gw2)
        gb2 = torch.from_numpy(gb2)

        w2 = w2[toKeep2_indices,:]
        w2 = w2[:,toKeep_indices]
        b2 = b2[toKeep2_indices]
        w2 = torch.from_numpy(w2)
        b2 = torch.from_numpy(b2)

        if(torch.cuda.is_available()):
            gw2 = gw2.cuda()
            w2 = w2.cuda()
            gb2 = gb2.cuda()
            b2 = b2.cuda()

        #net.convs[i] = nn.Linear(sum(toKeep_array), sum(toKeep2_array))
        net.convs[i].weight = torch.nn.Parameter(w2)
        net.convs[i].bias = torch.nn.Parameter(b2)
        net.convs[i].in_channels = len(w2[0])
        net.convs[i].out_channels = len(w2)
        net.convs[i].weight.grad = torch.nn.Parameter(gw2)
        net.convs[i].bias.grad = torch.nn.Parameter(gb2)
        net.bns[i] = nn.BatchNorm1d(len(net.convs[i].bias))
        toKeep_indices = toKeep2_indices
    fc1_w = net.fc1.weight.data.copy_(net.fc1.weight.data).cpu().numpy()
    fc1_wg = net.fc1.weight.grad.data.copy_(net.fc1.weight.grad.data).cpu().numpy()
    fc1_w = fc1_w[:,toKeep_indices]
    fc1_wg = fc1_wg[:,toKeep_indices]
    fc1_w = torch.from_numpy(fc1_w)
    fc1_wg = torch.from_numpy(fc1_wg)
    if (torch.cuda.is_available()):
        fc1_wg = fc1_wg.cuda()
        fc1_w = fc1_w.cuda()
    net.fc1.weight = torch.nn.Parameter(fc1_w)
    net.fc1.weight.grad = torch.nn.Parameter(fc1_wg)

    if(torch.cuda.is_available()):
        net = net.cuda()

    return net

def add_neurons(net, newNs, keep_grad = True, init="he", nclasses = 2):

    for i in range(len(newNs)):
        if (newNs[i] > 0):
            net.hebbs[i] = torch.cat((net.hebbs[i],Variable(torch.zeros(newNs[i]))))

    if(keep_grad == True and init == 'randn'):
        for j in range(len(newNs)):
            if (newNs[j] > 0):
                net.fcs[j].weight.data = torch.cat(
                    (net.fcs[j].weight.data, torch.mul(torch.randn([ newNs[j], len(net.fcs[j].weight.data[0])]), 0.1)), 0)
                net.fcs[j].bias.data = torch.cat((net.fcs[j].bias.data, torch.zeros(newNs[j])))
                net.fcs[j+1].weight.data = torch.cat(
                    (net.fcs[j+1].weight.data, torch.mul(torch.randn([len(net.fcs[j+1].weight.data), newNs[j]]), 0.1)), 1)
                net.fcs[j].weight.grad.data = torch.cat((net.fcs[j].weight.grad.data,torch.zeros([newNs[j],len(net.fcs[j].weight.data[0])])),0)
                net.fcs[j].bias.grad.data = torch.cat((net.fcs[j].bias.grad.data,torch.zeros(newNs[j])))
                net.fcs[j + 1].weight.grad.data = torch.cat((net.fcs[j].weight.grad.data,torch.zeros([len(net.fcs[j+1].weight.grad.data),newNs[j]])),1)

        net.fcs[-1].weight.data = torch.cat((net.fc3.weight.data, torch.mul(torch.randn([nclasses, newNs[-1]]), 0.1)), 1)
        net.fcs[-1].weight.grad.data = torch.cat((net.fc3.weight.grad.data,torch.zeros([nclasses,newNs[-1]])),1)

    elif(keep_grad == True and init == "he"):
        if (newNs[0] > 0):
            net.fc1.weight.data = torch.cat(
                (net.fc1.weight.data, torch.nn.init.kaiming_uniform(
                    torch.zeros([net.fc1.weight.data.shape[0] + newNs[0], net.fc1.weight.data.shape[1]]))[0:newNs[0]]), 0)
            net.fc1.bias.data = torch.cat((net.fc1.bias.data, torch.zeros(newNs[0])))
            net.fc1.weight.grad.data = torch.cat(
                (net.fc1.weight.grad.data, torch.nn.init.kaiming_uniform(
                    torch.zeros([net.fc1.weight.grad.data.shape[0] + newNs[0], net.fc1.weight.grad.data.shape[1]]))[0:newNs[0]]), 0)
            net.fc1.bias.grad.data = torch.cat((net.fc1.bias.grad.data, torch.zeros(newNs[0])))

            net.fc1.out_features = len(net.fc1.bias)

        for j in range(len(newNs)-1):
            if (newNs[j+1] > 0):
                net.fcs[j].bias.data = torch.cat((net.fcs[j].bias.data, torch.zeros(newNs[j+1])))
                net.fcs[j].weight.data = torch.cat(
                    (net.fcs[j].weight.data, torch.nn.init.kaiming_uniform(
                        torch.zeros([net.fcs[j].weight.data.shape[0],net.fcs[j].weight.data.shape[1]+newNs[j+1]]))[:,0:newNs[j+1]]), 1)
                net.fcs[j].weight.data = torch.cat(
                    (net.fcs[j].weight.data, torch.nn.init.kaiming_uniform(
                        torch.zeros([net.fcs[j].weight.data.shape[0]+newNs[j+1], net.fcs[j].weight.data.shape[1]]))[0:newNs[j+1],:]), 0)

                #net.fcs[j].weight.grad.data = torch.cat((net.fcs[j].weight.grad.data,torch.zeros([newNs[j+1],len(net.fcs[j].weight.data[0])])),0)
                net.fcs[j].bias.grad.data = torch.cat((net.fcs[j].bias.grad.data,torch.zeros(newNs[j+1])))
                net.fcs[j].weight.grad.data = torch.cat((net.fcs[j].weight.grad.data,
                                            torch.zeros([len(net.fcs[j].weight.grad.data),newNs[j+1]])),1)
                net.fcs[j].weight.grad.data = torch.cat((net.fcs[j].weight.grad.data,
                                            torch.zeros([newNs[j+1],len(net.fcs[j].weight.grad.data[0])])),0)
                net.fcs[j].out_features = len(net.fcs[j].weight.data)
                net.fcs[j].in_features = len(net.fcs[j].weight.data[0])

        if(sum(newNs) > 0):
            net.fclast.weight.data = torch.cat(
                    (net.fclast.weight.data, torch.nn.init.kaiming_uniform(
                        torch.zeros([net.fclast.weight.data.shape[0],net.fclast.weight.data.shape[1]+newNs[-1]]))[:,0:newNs[-1]]), 1)
            net.fclast.weight.grad.data = torch.cat((net.fclast.weight.grad.data,torch.zeros([nclasses,newNs[-1]])),1)
            net.fclast.in_features = len(net.fclast.weight.data[0])
    elif(keep_grad == False):
        if (newNs[0] > 0):

            net.fc1.weight = torch.nn.Parameter(torch.cat(
                (net.fc1.weight.data, torch.mul(torch.randn([newNs[0], len(net.fc1.weight.data[0])]), 0.1)), 0))
            net.fc1.bias = torch.nn.Parameter(torch.cat((net.fc1.bias.data, torch.zeros(newNs[0]))))
            net.fc2.weight = torch.nn.Parameter(torch.cat(
                (net.fc2.weight.data, torch.mul(torch.randn([len(net.fc2.weight.data), newNs[0]]), 0.1)), 1))
        if(newNs[1]>0):
            net.fc2.weight = torch.nn.Parameter(torch.cat(
                (net.fc2.weight.data, torch.mul(torch.randn([newNs[1], len(net.fc2.weight.data[0])]), 0.1)), 0))
            net.fc2.bias = torch.nn.Parameter(torch.cat((net.fc2.bias.data, torch.zeros(newNs[1]))))

            net.fc3.weight = torch.nn.Parameter(torch.cat((net.fc3.weight.data, torch.mul(torch.randn([nclasses, newNs[1]]), 0.1)), 1))

    elif(keep_grad == 'copy'):

        if (newNs[0] > 0):
            copies1 = torch.from_numpy(np.random.randint(0, len(net.fc1.bias.data) - 1, newNs[0]))
            net.fc1.weight.data = torch.cat((net.fc1.weight.data, net.fc1.weight.data[copies1]), 0)
            net.fc1.bias.data = torch.cat((net.fc1.bias.data, net.fc1.bias.data[copies1]))
            net.fc2.weight.data = torch.cat((net.fc2.weight.data, net.fc2.weight.data[:,copies1]), 1)

            net.fc1.weight.grad.data = torch.cat((net.fc1.weight.grad.data,net.fc1.weight.grad.data[copies1]),0)
            net.fc1.bias.grad.data = torch.cat((net.fc1.bias.grad.data,net.fc1.bias.grad.data[copies1]))
            net.fc2.weight.grad.data = torch.cat((net.fc2.weight.grad.data,net.fc2.weight.grad.data[:,copies1]),1)

        if (newNs[1] > 0):
            copies2 = torch.from_numpy(np.random.randint(0, len(net.fc2.bias.data) - 1, newNs[1]))
            net.fc2.weight.data = torch.cat( (net.fc2.weight.data, net.fc2.weight.data[copies2]), 0)
            net.fc2.bias.data = torch.cat((net.fc2.bias.data, net.fc2.bias.data[copies2]))
            net.fc3.weight.data = torch.cat((net.fc3.weight.data, net.fc3.weight.data[:,copies2]), 1)
            net.fc2.weight.grad.data = torch.cat((net.fc2.weight.grad.data,net.fc2.weight.grad.data[copies2]),0)
            net.fc2.bias.grad.data = torch.cat((net.fc2.bias.grad.data,net.fc2.bias.grad.data[copies2]))
            net.fc3.weight.grad.data = torch.cat((net.fc3.weight.grad.data,net.fc3.weight.grad.data[:,copies2]),1)
    else:
        print("ERROR")
    for i in range(len(net.hebbs)):
        net.bns[i] = nn.BatchNorm1d(len(net.hebbs[i]))

    return net

def add_conv_units(net, new_conv_channels=[4,8,16,32,64,128], keep_grad = True, init="he", nclasses = 2, clipmax = 100000):
    # TODO augment by a factor, e.g. x2. like that the archtecture would be kept (unless debalanced by the pruning process). Or would be fed like this: [2,4,8,16,...]

    for i in range(len(new_conv_channels)):
        if (new_conv_channels[i] > 0):
            net.hebbs_conv[i] = torch.cat((net.hebbs_conv[i],Variable(torch.zeros(new_conv_channels[i]))))

    if(keep_grad == True and init == "he"):
        if (new_conv_channels[0] > 0 and  len(net.convs[0].bias) <= clipmax):
            print("New neurons with kaiming init")
            net.convs[0].weight.data = torch.cat(
                (net.convs[0].weight.data, torch.nn.init.kaiming_uniform(
                    torch.zeros([net.convs[0].weight.data.shape[0] + new_conv_channels[0], net.convs[0].weight.data.shape[1],
                        net.convs[0].weight.data.shape[2], net.convs[0].weight.data.shape[3]]))[0:new_conv_channels[0]]), 0)

            net.convs[0].bias.data = torch.cat((net.convs[0].bias.data, torch.zeros(new_conv_channels[0])))

            net.convs[0].weight.grad.data = torch.cat(
                (net.convs[0].weight.grad.data, torch.nn.init.kaiming_uniform(
                    torch.zeros([net.convs[0].weight.grad.data.shape[0] + new_conv_channels[0],
                                 net.convs[0].weight.grad.data.shape[1],net.convs[0].weight.grad.data.shape[2],
                                 net.convs[0].weight.grad.data.shape[3]]))[0:new_conv_channels[0]]), 0)

            net.convs[0].bias.grad.data = torch.cat((net.convs[0].bias.grad.data, torch.zeros(new_conv_channels[0])))

            net.convs[0].out_channels = len(net.convs[0].bias)
            net.planes[1] = len(net.convs[0].bias.data)

        for j in range(1,len(new_conv_channels)):
            if (new_conv_channels[j] > 0 and len(net.convs[j].bias) < clipmax):
                print("New neurons with kaiming init")
                net.convs[j].bias.data = torch.cat((net.convs[j].bias.data, torch.zeros(new_conv_channels[j])))
                net.convs[j].weight.data = torch.cat(
                    (net.convs[j].weight.data, torch.nn.init.kaiming_uniform(
                        torch.zeros([net.convs[j].weight.data.shape[0],net.convs[j].weight.data.shape[1]+new_conv_channels[j-1],
                        net.convs[j].weight.grad.data.shape[2],net.convs[j].weight.grad.data.shape[3]]))[:,0:new_conv_channels[j-1]]), 1)
                net.convs[j].weight.data = torch.cat(
                    (net.convs[j].weight.data, torch.nn.init.kaiming_uniform(
                        torch.zeros([net.convs[j].weight.data.shape[0]+new_conv_channels[j], net.convs[j].weight.data.shape[1],
                        net.convs[j].weight.data.shape[2],net.convs[j].weight.data.shape[3]]))[0:new_conv_channels[j],:]), 0)

                #net.convs[j].weight.grad.data = torch.cat((net.convs[j].weight.grad.data,torch.zeros([new_conv_channels[j],len(net.convs[j].weight.data[0])])),0)
                net.convs[j].bias.grad.data = torch.cat((net.convs[j].bias.grad.data,torch.zeros(new_conv_channels[j])))
                net.convs[j].weight.grad.data = torch.cat((net.convs[j].weight.grad.data,
                        torch.zeros([net.convs[j].weight.grad.data.shape[0],new_conv_channels[j-1],
                        net.convs[j].weight.grad.data.shape[2],net.convs[j].weight.grad.data.shape[3]])),1)
                net.convs[j].weight.grad.data = torch.cat((net.convs[j].weight.grad.data,
                        torch.zeros([new_conv_channels[j],net.convs[j].weight.grad.data.shape[1],
                        net.convs[j].weight.grad.data.shape[2],net.convs[j].weight.grad.data.shape[3]])),0)
                net.convs[j].out_channels = len(net.convs[j].weight.data)
                net.convs[j].in_channels = len(net.convs[j].weight.data[0])
                net.planes[j+1] = len(net.convs[j].bias.data)

            else:
                print("Already the max neurons. Put them on another layer or place new layer")
        #Need to modifiy the number of inputs in fc1

        net.fc1.in_features = net.planes[-1]
        net.fc1.weight.data = torch.cat(
            (net.fc1.weight.data, torch.nn.init.kaiming_uniform(
                torch.zeros([net.fc1.weight.data.shape[0], net.planes[-1]]))[:,
                                     0:new_conv_channels[-1]]), 1)
        net.fc1.weight.grad.data = torch.cat(
            (net.fc1.weight.grad.data, torch.nn.init.kaiming_uniform(
                torch.zeros([net.fc1.weight.grad.data.shape[0], net.planes[-1]]))[:,
                                     0:new_conv_channels[-1]]), 1)

    elif(keep_grad == 'copy'):

        if (new_conv_channels[0] > 0):
            copies1 = torch.from_numpy(np.random.randint(0, len(net.fc1.bias.data) - 1, new_conv_channels[0]))
            net.fc1.weight.data = torch.cat((net.fc1.weight.data, net.fc1.weight.data[copies1]), 0)
            net.fc1.bias.data = torch.cat((net.fc1.bias.data, net.fc1.bias.data[copies1]))
            net.fc2.weight.data = torch.cat((net.fc2.weight.data, net.fc2.weight.data[:,copies1]), 1)

            net.fc1.weight.grad.data = torch.cat((net.fc1.weight.grad.data,net.fc1.weight.grad.data[copies1]),0)
            net.fc1.bias.grad.data = torch.cat((net.fc1.bias.grad.data,net.fc1.bias.grad.data[copies1]))
            net.fc2.weight.grad.data = torch.cat((net.fc2.weight.grad.data,net.fc2.weight.grad.data[:,copies1]),1)

        if (new_conv_channels[1] > 0):
            copies2 = torch.from_numpy(np.random.randint(0, len(net.fc2.bias.data) - 1, new_conv_channels[1]))
            net.fc2.weight.data = torch.cat( (net.fc2.weight.data, net.fc2.weight.data[copies2]), 0)
            net.fc2.bias.data = torch.cat((net.fc2.bias.data, net.fc2.bias.data[copies2]))
            net.fc3.weight.data = torch.cat((net.fc3.weight.data, net.fc3.weight.data[:,copies2]), 1)
            net.fc2.weight.grad.data = torch.cat((net.fc2.weight.grad.data,net.fc2.weight.grad.data[copies2]),0)
            net.fc2.bias.grad.data = torch.cat((net.fc2.bias.grad.data,net.fc2.bias.grad.data[copies2]))
            net.fc3.weight.grad.data = torch.cat((net.fc3.weight.grad.data,net.fc3.weight.grad.data[:,copies2]),1)
    else:
        print("ERROR")
    for i in range(len(net.hebbs)):
        net.bns[i] = nn.BatchNorm1d(len(net.hebbs[i]))

    return net

# A way to regularize the network by replacing the neurons instead of adding or removing. So input size == output size
def replace_neurons(net,n):
    pass







