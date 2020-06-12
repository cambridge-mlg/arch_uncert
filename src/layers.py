from __future__ import division
import torch.nn as nn

# More general residual layers that take probabilities as method inputs


class global_mean_pool_2d(nn.Module):
    def __init__(self):
        super(global_mean_pool_2d, self).__init__()

    def forward(self, x):
        return x.mean(dim=(2,3))


class res_MLPBlock(nn.Module):
    def __init__(self, width):
        super(res_MLPBlock, self).__init__()
        self.ops = nn.Sequential(nn.Linear(width, width), nn.ReLU(),  nn.BatchNorm1d(num_features=width))

    def forward(self, x):
        return x + self.ops(x)


class bern_MLPBlock(nn.Module):
    """Skippable MLPBlock with relu"""
    def __init__(self, width):
        super(bern_MLPBlock, self).__init__()

        self.block = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.BatchNorm1d(num_features=width))

    def forward(self, x, b):
        """b is sample from binary variable or activation probability (soft forward)"""
        return x + b * self.block(x)


class bern_bottleneck_convBlock(nn.Module):
    """Skippable bottleneck convolutional preactivation Resnet Block"""
    def __init__(self, inner_dim, outer_dim):
        super(bern_bottleneck_convBlock, self).__init__()

        self.block = self.net = nn.Sequential(
            nn.BatchNorm2d(outer_dim),
            nn.ReLU(),
            nn.Conv2d(outer_dim, inner_dim, 1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
            nn.Conv2d(inner_dim, outer_dim, 1),
        )

    def forward(self, x, b):
        return x + b * self.block(x)

