from __future__ import division
import torch
import torch.nn as nn
from src.layers import bern_MLPBlock, bern_bottleneck_convBlock, global_mean_pool_2d, res_MLPBlock


class arq_uncert_fc_resnet(nn.Module):
    """Class for fc variational architecture resnet with new more modular structure"""
    def __init__(self, input_dim, output_dim, width, n_layers, prob_model):
        super(arq_uncert_fc_resnet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = nn.Linear(self.input_dim, width)
        self.output_layer = nn.Linear(width, self.output_dim)
        self.n_layers = n_layers
        self.width = width

        self.prob_model = prob_model

        stochstic_layers = []
        for i in range(self.n_layers):
            stochstic_layers.append(bern_MLPBlock(width))
        self.stochstic_layers = nn.Sequential(*stochstic_layers)

    def get_q_vector(self):
        return self.prob_model.get_mask_probs()

    def get_KL(self):
        return self.prob_model.get_KL()

    def forward(self, x):
        return self.forward_get_acts(x, depth=None)

    def vec_forward(self, x, vec):
        assert vec.shape[0] == self.n_layers
        x = self.input_layer(x)
        for i in range(self.n_layers):
            x = self.stochstic_layers[i](x, vec[i])
        x = self.output_layer(x)
        return x

    def forward_get_acts(self, x):
        # TODO: prealocate a zero vector of the same type as x
        act_vec = []
        x = self.input_layer(x)
        act_vec.append(self.output_layer(x).unsqueeze(0))
        for i in range(self.n_layers):
            x = self.stochstic_layers[i](x, 1)
            act_vec.append(self.output_layer(x).unsqueeze(0))
        act_vec = torch.cat(act_vec, dim=0)
        return act_vec


class arq_uncert_conv2d_resnet(nn.Module):
    """Class for convolutional variational architecture resnet with new more modular structure"""
    def __init__(self, input_chan, output_dim, outer_width, inner_width, n_layers, prob_model):
        super(arq_uncert_conv2d_resnet, self).__init__()

        self.input_chan = input_chan
        self.output_dim = output_dim
        self.outer_width = outer_width
        self.inner_width = inner_width
        self.input_layer = nn.Sequential(nn.Conv2d(self.input_chan, outer_width, 5), nn.AvgPool2d(kernel_size=(2,2)))
        self.output_layer = nn.Sequential(global_mean_pool_2d(), res_MLPBlock(outer_width), nn.Linear(outer_width, self.output_dim))
        self.n_downsample_layer = nn.AvgPool2d(kernel_size=(2,2))
        self.n_layers = n_layers

        self.prob_model = prob_model

        stochstic_layers = []
        for i in range(self.n_layers):
            stochstic_layers.append(bern_bottleneck_convBlock(inner_width, outer_width))
        self.stochstic_layers = nn.Sequential(*stochstic_layers)

    def get_q_vector(self):
        return self.prob_model.get_mask_probs()

    def get_KL(self):
        return self.prob_model.get_KL()

    def forward(self, x):
        return self.forward_get_acts(x, depth=None)

    def vec_forward(self, x, vec):
        assert vec.shape[0] == self.n_layers

        x = self.input_layer(x)
        for i in range(self.n_layers):
            x = self.stochstic_layers[i](x, vec[i])
        x = self.output_layer(x)
        return x

    def forward_get_acts(self, x, depth=None):
        act_vec = []
        x = self.input_layer(x)
        act_vec.append(self.output_layer(x).unsqueeze(0))

        n_layers = depth if depth is not None else self.n_layers

        for i in range(n_layers):
            x = self.stochstic_layers[i](x, 1)
            act_vec.append(self.output_layer(x).unsqueeze(0))
        act_vec = torch.cat(act_vec, dim=0)
        return act_vec
