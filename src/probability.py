from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical
import numpy as np
from src.utils import torch_onehot



class variational_categorical(nn.Module):
    """Class for variational inference with categorical approx posterior and prior
    dims should be set to N_layers but there are layers+1 categories in order to allow for the linear model (0 res blocks)"""
    def __init__(self, dims, prior_probs, temp=0.1, eps=1e-10, cuda=True):
        super(variational_categorical, self).__init__()

        self.mask_mtx = torch.cat([torch.zeros(1, dims), torch.ones((dims, dims)).tril()], dim=0)
        self.dims = dims + 1  # dim [0] will refer to 0 layers active: linear explanation

        self.q_logits = nn.Parameter(torch.zeros(self.dims), requires_grad=True)

        self.prior_probs = torch.Tensor(prior_probs)
        if self.prior_probs.shape[0] == 1:
            self.prior_probs = self.prior_probs.expand(self.dims)
        assert self.prior_probs.shape[0] == self.dims
        assert self.prior_probs.sum().item() - 1 < 1e-6

        self.eps = eps
        self.temp = temp
        self.cuda = cuda
        if self.cuda:
            self.to_cuda()

        self.name = 'cat'
        self.summary = ({'prior_probs': prior_probs, 'temp': temp})

    def to_cuda(self):
        self.prior_probs = self.prior_probs.cuda()
        self.q_logits.data = self.q_logits.data.cuda()

    def get_q_probs(self):
        """Get probs of each depth configuration"""
        return F.softmax(self.q_logits, dim=0)

    def get_mask_probs(self):
        """Get probs of each layer being on individually (usefull for soft forward pass)"""
        return torch.cumsum(self.get_q_probs().flip(dims=[0]), dim=0).flip(dims=[0])[1:]  # We start at 1 as the prob of linear model is implicit


    def get_KL(self):
        """KL between categorical distributions"""
        q = self.get_q_probs().clamp(min=self.eps, max=(1 - self.eps))
        p = self.prior_probs.clamp(min=self.eps, max=(1 - self.eps))
        KL = (q * (torch.log(q) - torch.log(p))).sum()
        return KL


    def efficient_E_loglike(self, act_vec, y, f_neg_loglike):
        """Calculate ELBO with deterministic expectation."""
        batch_size = act_vec.shape[1]
        depth = act_vec.shape[0]
        q = self.get_q_probs()
        q_expand = q.repeat_interleave(batch_size, dim=0)  # Repeat to match batch_size
        y_expand = y.repeat(depth)  # targets are same across acts -> interleave

        act_vec_flat = act_vec.view(depth*batch_size, -1)  # flattening results in batch_n changing first

        neg_loglike_per_act = f_neg_loglike(act_vec_flat, y_expand, reduction='none')
        mean_neg_loglike = (neg_loglike_per_act * q_expand).view(depth, batch_size).sum(dim=0).mean()
        return mean_neg_loglike

    def efficient_predict(self, act_vec, softmax=False):
        if softmax:
            preds = F.softmax(act_vec, dim=2)
        else:
            preds = act_vec
        q = self.get_q_probs()
        while len(q.shape) < len(act_vec.shape):
            q = q.unsqueeze(1)
        weighed_preds = q * preds
        return weighed_preds

    def efficient_predict_d(self, act_vec, depth, softmax=False):
        if softmax:
            preds = F.softmax(act_vec, dim=2)
        else:
            preds = act_vec
        preds = preds[:depth+1]
        q = self.get_q_probs()
        q[depth] += q[depth+1:].sum()
        q = q[:depth+1]
        while len(q.shape) < len(act_vec.shape):
            q = q.unsqueeze(1)
        weighed_preds = q * preds
        return weighed_preds


class fixed_probs():
    """Class that fixes the activation probability of all layers."""
    def __init__(self, dims, probs=[1.], distribution_name=None, cuda=True):
        super(fixed_probs, self).__init__()

        self.mask_mtx = torch.cat([torch.zeros(1, dims), torch.ones((dims, dims)).tril()], dim=0)
        dims = dims + 1  # dims originally = number of skip-layers

        self.mask_probs = torch.Tensor(probs)
        if self.mask_probs.shape[0] == 1:
            self.mask_probs = self.mask_probs.expand(dims)
        self.cuda = cuda
        if self.cuda:
            self.mask_probs = self.mask_probs.cuda()

        self.distribution_name = distribution_name
        if self.distribution_name == 'cat' or self.distribution_name == 'categorical':
            self.distribution = Categorical(self.mask_probs)
            self.name = 'deterministic_cat'
            self.summary = ({'probs': probs})
        elif self.distribution_name == 'bern' or self.distribution_name == 'bernouilli':
            self.distribution = Bernoulli(self.mask_probs)
            self.name = 'deterministic_bernouilli'
            self.summary = ({'probs': probs})
        else:
            self.distribution = None


    def get_mask_probs(self):
        if self.distribution_name == 'cat' or self.distribution_name == 'categorical':
            return torch.cumsum(self.mask_probs.flip(dims=[0]), dim=0).flip(dims=[0])[1:]
        else:
            return self.mask_probs

    def get_q_probs(self):
        """Get probs of each depth configuration"""
        return self.mask_probs

    def get_KL(self):
        if self.cuda:
            return torch.Tensor([0]).cuda()
        else:
            return torch.Tensor([0])

    def efficient_E_loglike(self, act_vec, y, f_neg_loglike):
        """Calculate ELBO with deterministic expectation."""
        batch_size = act_vec.shape[1]
        depth = act_vec.shape[0]
        q = self.mask_probs
        q_expand = q.repeat_interleave(batch_size, dim=0)  # Repeat to match batch_size
        y_expand = y.repeat(depth)  # targets are same across acts -> interleave

        act_vec_flat = act_vec.view(depth*batch_size, -1)  # flattening results in batch_n changing first

        neg_loglike_per_act = f_neg_loglike(act_vec_flat, y_expand, reduction='none')
        mean_neg_loglike = (neg_loglike_per_act * q_expand).view(depth, batch_size).sum(dim=0).mean()
        return mean_neg_loglike

    def efficient_predict(self, act_vec, softmax=False):
        if softmax:
            preds = F.softmax(act_vec, dim=2)
        else:
            preds = act_vec
        q = self.mask_probs
        while len(q.shape) < len(act_vec.shape):
            q = q.unsqueeze(1)
        weighed_preds = q * preds
        return weighed_preds

