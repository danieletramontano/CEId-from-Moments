"""
Functions to implement the Graphical RICA method.
"""
import torch
import numpy as np
import networkx as nx
from torch.nn import Parameter




def intersection(lst1, lst2):
    """
    Compute the intersection of two lists.
    """
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def count_lists(l):
    """
    Count the number of zeros and non-zeros in a list.
    """
    c_m = -1
    c_um = -1
    c_list = np.zeros(len(l))
    for i, _ in enumerate(l):
        if l[i] == 0:
            c_um = c_um+1
            c_list[i] = c_um
        else:
            c_m = c_m+1
            c_list[i] = c_m
    return c_list


# Remark: data - before withening
def init_w_guess_(data, g, latent):
    up_data = data.t()
    w = torch.zeros(len(g.edges()))
    mask = torch.zeros(len(g.edges()))

    for i, e in enumerate(g.edges()):
        if e[0] < latent:
            w[i] = torch.Tensor(1).normal_().item()
            mask[i] = 1
        else:
            G_cov = up_data.cov()
            w[i] = G_cov[e[0]-latent,e[1]-latent]/G_cov[e[0]-latent,e[0]-latent]
            up_data[:, e[1]-latent] = up_data[:, e[1]-latent]-w[i]*up_data[:,e[0]-latent]

            an_s = sorted(nx.ancestors(g, e[0]))
            i_s = intersection(an_s,list(range(latent)))

            if len(i_s) > 0 :
                an_t = sorted(nx.ancestors(g, e[1]))
                i_t = intersection(an_t,list(range(latent)))
                if len(i_t)>0:
                    ints = intersection(i_s,i_t)
                    if len(ints)>0:
                        mask[i] = 1
    return w, mask




def graphical_rica(latent, observed, g, data, data_whitening, epochs, lr, W_w, w_init, momentum=0, lmbda=0):

    """
        Graphical adaptation of RICA

        Parameters:
            - latent (int): Number of hidden variables.
            - observed (int): Number of observed variables.
            - g (nx.DiGraph): The DAG as a NetworkX DiGraph object.
            - data (torch.Tensor): Input data.
            - lr(double): Learning rate of the optimizer.
            - epochs (int): Number of optimization epochs.
            - W_w (torch.Tensor): Whitening matrix.
            - w_init (str): Weight initialization strategy ('random', 'true', 'cov_guess').

        Returns:
            - weight_pred (torch.Tensor): The learned weights.
        """

    loss_data = torch.zeros(epochs)

    mask = None
    if w_init=='cov_guess':
        w, mask = init_w_guess_(data, g, latent)
        weight = Parameter(w[mask==1])
        fix_weight = w[mask == 0]
        c_list = count_lists(mask)
    else:
        weight = Parameter(torch.Tensor(len(g.edges())).normal_(0,1))

    optimizer = torch.optim.RMSprop([weight], lr, momentum=momentum)

    min_loss = None
    for epoch in range(epochs):
        adj = torch.eye(len(g.nodes()))
        if w_init == 'cov_guess':
            for ii, e in enumerate(g.edges()):
                if mask[ii] == 1:
                    adj[e] = -weight[int(c_list[ii])]
                else:
                    adj[e] = -fix_weight[int(c_list[ii])]
        else:
            for e in range(len(g.edges())):
                adj[list(g.edges)[e]]=-weight[e]

        B = (torch.inverse(adj)).t()
        B = B[latent:latent+observed,:]
        B = W_w.matmul(B)

        latents = data_whitening.matmul(B)
        output = latents.matmul(B.t())

        diff = output - data_whitening
        loss_recon = 0
        if lmbda!=0:
            loss_recon = (diff * diff).mean()
        loss_latent = latents.abs().mean()
        loss = lmbda * loss_recon + loss_latent

        loss_data[epoch] = (loss.data).item()
        if min_loss is None or min_loss > loss_data[epoch]:
            min_loss = loss_data[epoch]
            weight_pred = weight.detach().clone()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return weight_pred