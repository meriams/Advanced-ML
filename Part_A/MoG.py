# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm


class GaussianMixturePrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        
        K: [int]
           Number of components in the mixture.

        """
        super(GaussianMixturePrior, self).__init__()
        self.M = M
        self.K = K
        self.means = nn.Parameter(torch.randn(self.K, self.M))
        self.stds = nn.Parameter(torch.ones(self.K, self.M))
        self.weights = nn.Parameter(torch.ones(self.K) / self.K)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        mix = td.Categorical(self.weights)
        comp = td.Independent(td.Normal(self.means, self.stds), 1)
        return td.MixtureSameFamily(mix, comp)
    

class MoGEncoder(nn.Module):
    def __init__(self, encoder_net, K, M):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(MoGEncoder, self).__init__()
        self.encoder_net = encoder_net
        self.num_components = K
        self.latent_dim = M

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        output = self.encoder_net(x)
        mean, log_std = torch.chunk(output, 2, dim=-1)
        
        # Reshape mean and log_std to match the shape of the MoG prior
        mean = mean.view(-1, self.num_components, self.latent_dim)
        log_std = log_std.view(-1, self.num_components, self.latent_dim)

        # Transform log_std to std
        std = torch.exp(log_std)

        # Create a mixture of Gaussians distribution
        mixture = td.Normal(mean, std)
        mix_weights = nn.functional.softmax(torch.randn(output.shape[0], self.num_components), dim=-1)
        return td.MixtureSameFamily(td.Categorical(mix_weights), mixture)


