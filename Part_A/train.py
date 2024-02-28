import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
from vae import *
from Gaussain_prior import *
from MoG import *
from vae_MoG import *
import matplotlib.pyplot as plt
import numpy as np


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

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

def test_log_likelihood(model, data_loader, device):
    model.eval()

    total_elbo = 0.0
    total_samples = 0

    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            elbo = model.elbo(x)
            total_elbo += torch.sum(elbo).item()
            total_samples += x.size(0)

    test_ll = total_elbo / total_samples

    return test_ll


def plot_prior_and_posterior(prior, encoder, data_loader, device, save_path):
    # Sample from prior
    prior_samples = prior().sample(torch.Size([len(data_loader.dataset)])).cpu()

    # Sample from aggregate posterior
    encoder.eval()
    posterior_samples = []
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            q = encoder(x)
            z = q.sample()
            posterior_samples.append(z.cpu())
    posterior_samples = torch.cat(posterior_samples, dim=0)

    # Plot PDFs
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute histograms
    prior_hist, prior_bins = np.histogram(prior_samples[:, 0].numpy(), bins=30, density=True)
    posterior_hist, posterior_bins = np.histogram(posterior_samples[:, 0].numpy(), bins=30, density=True)

    # Compute bin centers
    prior_bin_centers = (prior_bins[:-1] + prior_bins[1:]) / 2
    posterior_bin_centers = (posterior_bins[:-1] + posterior_bins[1:]) / 2

    # Plot PDFs
    ax.plot(prior_bin_centers, prior_hist, color='blue', label='Prior', linewidth=2)
    ax.plot(posterior_bin_centers, posterior_hist, color='green', label='Posterior', linewidth=2)

    ax.set_title('Prior and Aggregate Posterior Distributions')
    ax.set_xlabel('Latent Variable')
    ax.set_ylabel('Probability Density')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


"""
def plot_prior_and_posterior(prior, encoder, data_loader, device, save_path):
   
    # Sample from prior
    prior_samples = prior().sample(torch.Size([len(data_loader.dataset)])).cpu()

    # Sample from aggregate posterior
    encoder.eval()
    posterior_samples = []
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            q = encoder(x)
            z = q.sample()
            posterior_samples.append(z.cpu())
    posterior_samples = torch.cat(posterior_samples, dim=0)

    # Plot histograms
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(prior_samples[:, 0], bins=30, color='blue', alpha=0.7, label='Prior')
    axs[0].set_title('Prior Distribution')
    axs[0].set_xlabel('Latent Variable')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    axs[1].hist(posterior_samples[:, 0], bins=30, color='green', alpha=0.7, label='Posterior')
    axs[1].set_title('Aggregate Posterior Distribution')
    axs[1].set_xlabel('Latent Variable')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
"""




if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    #K = 10
    prior = GaussianPrior(M)
    #prior = GaussianMixturePrior(M, K)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
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
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    #encoder = MoGEncoder(encoder_net, K, M)
    #model = VAE_MoG(prior, decoder, encoder).to(device)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), 'model1.pt')#args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load('model1.pt', map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(4)).cpu() 
            save_image(samples.view(4, 1, 28, 28), 'samples1.png') #args.samples)

        # Evaluate test set log-likelihood
        test_ll = test_log_likelihood(model, mnist_test_loader, device)
        print("Test set log-likelihood: {:.2f}".format(test_ll))

        # Save prior and aggregate posterior plot
        plot_save_path = 'prior_posterior_plot1.png'
        plot_prior_and_posterior(prior, encoder, mnist_train_loader, device, plot_save_path)
