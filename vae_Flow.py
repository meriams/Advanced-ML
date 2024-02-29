# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.1 (2024-01-29)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm


class GaussianBase(nn.Module):
    def __init__(self, D):
        """
        Define a Gaussian base distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the base distribution.
        """
        super(GaussianBase, self).__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.D), requires_grad=False)

    def forward(self):
        """
        Return the base distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MaskedCouplingLayer(nn.Module):
    """
    An affine coupling layer for a normalizing flow.
    """

    def __init__(self, scale_net, translation_net, mask):
        """
        Define a coupling layer.

        Parameters:
        scale_net: [torch.nn.Module]
            The scaling network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        translation_net: [torch.nn.Module]
            The translation network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        mask: [torch.Tensor]
            A binary mask of dimension `(feature_dim,)` that determines which features (where the mask is zero) are transformed by the scaling and translation networks.
        """
        super(MaskedCouplingLayer, self).__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        """
        Transform a batch of data through the coupling layer (from the base to data).

        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations of dimension `(batch_size, feature_dim)`.
        """

        # Apply mask
        z_masked = z * self.mask
        # Compute scaling and translation parameters
        log_s = self.scale_net(z_masked) * (1 - self.mask)
        t = self.translation_net(z_masked) * (1 - self.mask)
        # Transform
        z_transformed = z_masked + (1 - self.mask) * (z * torch.exp(log_s) + t)
        # Compute log determinant
        log_det_J = torch.sum(log_s, dim=1)
        return z_transformed, log_det_J

    def inverse(self, x):
        """
        Transform a batch of data through the coupling layer (from data to the base).

        Parameters:
        z: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        x: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        # Apply mask
        z_masked = x * self.mask
        # Compute scaling and translation parameters
        log_s = self.scale_net(z_masked) * (1 - self.mask)
        t = self.translation_net(z_masked) * (1 - self.mask)
        # Inverse transform
        z_original = z_masked + (1 - self.mask) * ((x - t) * torch.exp(-log_s))
        # Compute log determinant (negative of forward)
        log_det_J = -torch.sum(log_s, dim=1)
        return z_original, log_det_J


class Flow(nn.Module):
    def __init__(self, base, transformations):
        """
        Define a normalizing flow model.

        Parameters:
        base: [torch.distributions.Distribution]
            The base distribution.
        transformations: [list of torch.nn.Module]
            A list of transformations to apply to the base distribution.
        """
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z):
        """
        Transform a batch of data through the flow (from the base to data).

        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations.
        """
        sum_log_det_J = 0
        for T in self.transformations:
            x, log_det_J = T(z)
            sum_log_det_J += log_det_J
            z = x
        return x, sum_log_det_J

    def inverse(self, x):
        """
        Transform a batch of data through the flow (from data to the base).

        Parameters:
        x: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        sum_log_det_J = 0
        for T in reversed(self.transformations):
            z, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
            x = z
        return z, sum_log_det_J

    def log_prob(self, x):
        """
        Compute the log probability of a batch of data under the flow.

        Parameters:
        x: [torch.Tensor]
            The data of dimension `(batch_size, feature_dim)`
        Returns:
        log_prob: [torch.Tensor]
            The log probability of the data under the flow.
        """
        z, log_det_J = self.inverse(x)
        return self.base().log_prob(z) + log_det_J

    def sample(self, sample_shape=(1,)):
        """
        Sample from the flow.

        Parameters:
        n_samples: [int]
            Number of samples to generate.
        Returns:
        z: [torch.Tensor]
            The samples of dimension `(n_samples, feature_dim)`
        """
        z = self.base().sample(sample_shape)
        return self.forward(z)[0]

    def loss(self, x):
        """
        Compute the negative mean log likelihood for the given data bath.

        Parameters:
        x: [torch.Tensor]
            A tensor of dimension `(batch_size, feature_dim)`
        Returns:
        loss: [torch.Tensor]
            The negative mean log likelihood for the given data batch.
        """
        return -torch.mean(self.log_prob(x))


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        x = x.view(-1, 28, 28)
        log_p_z = self.prior.log_prob(z)
        log_q_z_x = q.log_prob(z)
        log_p_x_z = self.decoder(z).log_prob(x)

        elbo = torch.mean(log_p_x_z + log_p_z - log_q_z_x, dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior.sample(sample_shape=torch.Size([n_samples]))  # Adjusted line
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()
    num_steps = len(data_loader) * epochs
    epoch = 0

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            x = next(iter(data_loader))[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Report
            if step % 5 == 0:
                loss = loss.detach().cpu()
                pbar.set_description(f"epoch={epoch}, step={step}, loss={loss:.1f}")

            if (step + 1) % len(data_loader) == 0:
                epoch += 1


def plot_prior_and_posterior(prior, encoder, data_loader, device, save_path):
    import numpy as np

    # Sample from prior
    prior_samples = prior.sample(torch.Size([len(data_loader.dataset)])).cpu()

    # Sample from aggregate posterior
    encoder.eval()
    posterior_samples = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            q = encoder(x)
            z = q.rsample()
            posterior_samples.append(z.cpu())
    posterior_samples = torch.cat(posterior_samples, dim=0)

    # Plot PDFs
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute histograms
    prior_samples = prior_samples.detach()
    prior_hist, prior_bins = np.histogram(prior_samples[:, 0].numpy(), bins=30, density=True)
    posterior_hist, posterior_bins = np.histogram(posterior_samples[:, 0].numpy(), bins=30, density=True)

    # Compute bin centers
    prior_bin_centers = (prior_bins[:-1] + prior_bins[1:]) / 2
    posterior_bin_centers = (posterior_bins[:-1] + posterior_bins[1:]) / 2

    # Plot PDFs
    ax.plot(prior_bin_centers, prior_hist, color='blue', label='Prior', linewidth=2)
    ax.plot(posterior_bin_centers, posterior_hist, color='green', label='Posterior', linewidth=2)

    ax.set_title('Prior and Aggregate Posterior Distributions with Flow prior')
    ax.set_xlabel('Latent Variable')
    ax.set_ylabel('Probability Density')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'ELBO', 'Plot', 'sample_save'],
                        help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt',
                        help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png',
                        help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N',
                        help='dimension of latent variable (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze()),
            transforms.Lambda(lambda x: x.flatten())])), batch_size=32, shuffle=True)

    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze()),
            transforms.Lambda(lambda x: x.flatten())])), batch_size=32, shuffle=True)


    # Define prior distribution
    M = args.latent_dim

    num_transformations = 6
    num_hidden = 10

    transformations = []
    mask = torch.ones(M)
    mask[::2] = 0

    for i in range(num_transformations):
        mask = (1 - mask)  # Flip the mask
        scale_net = nn.Sequential(nn.Linear(M, num_hidden), nn.ReLU(), nn.BatchNorm1d(num_hidden),
                                  nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.BatchNorm1d(num_hidden),
                                  nn.Linear(num_hidden, M), nn.Tanh())

        translation_net = nn.Sequential(nn.Linear(M, num_hidden), nn.ReLU(), nn.BatchNorm1d(num_hidden),
                                        nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.BatchNorm1d(num_hidden),
                                        nn.Linear(num_hidden, M))

        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))


    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M * 2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    prior = Flow(GaussianBase(M), transformations)
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt

        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample((16))).cpu()
            samples = samples.view(-1, 28, 28)

            fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))

            for ax, img in zip(axes.flat, samples):
                ax.imshow(img, cmap='gray')

            plt.suptitle("Generated samples")
            plt.show()

    elif args.mode == 'ELBO':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        elbo_values = []

        model.eval()
        for batch_idx, (data, _) in enumerate(mnist_test_loader):
            data = data.to(device)
            with torch.no_grad():
                elbo = model.elbo(data).item()
            elbo_values.append(elbo)

        mean_elbo = sum(elbo_values) / len(elbo_values)

        print("Mean ELBO on the binarized MNIST test set:", mean_elbo)

    elif args.mode == 'Plot':
        import matplotlib.pyplot as plt

        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        plot_save_path = 'prior_posterior_plot1.png'
        plot_prior_and_posterior(prior, encoder, mnist_train_loader, device, plot_save_path)

    elif args.mode == 'sample_save':
        import os
        import matplotlib.pyplot as plt
        import numpy as np

        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = model.sample((4)).cpu()
            samples = samples.view(-1, 28, 28)

            if not os.path.exists('samples'):
                os.makedirs('samples')

            for i, img in enumerate(samples):
                img = (img >= 0.5).float()
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.savefig(f'samples/Flow_VAE_{i+1}.png')
                plt.close()
