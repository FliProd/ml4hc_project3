
from cProfile import label
from tabnanny import check
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.beta_vae import BetaVAE_CLF
from config import config

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)

    return total_kld 



def train_beta_vae(train_dataset, options):
    epochs = 1
    interval = 100
    batch_size = options['batch_size']
    reduction = options['loss_reduction']
    dataset_size = len(train_dataset)

    beta = options['beta']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('training on', device)

    model = BetaVAE_CLF(z_dim=options['z_dim'], nc=3, batch_size=batch_size,
                classifier=options['classifier'], 
                classifier_options=options['classifier_options'][options['classifier']]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr =0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)


    (loaded, Z, y) = load_model(options['model_identifier'], options['saved_model_path'], model, optimizer)
    if not loaded:

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)


        for e in range(epochs):
            # store latent representations to train classifier
            if e == (epochs - 1):
                Z = np.empty((dataset_size, options['z_dim']))
                y = np.empty((dataset_size))
            for i, (imgs, labels) in enumerate(train_dataloader):
                imgs = imgs.to(device)
                recon_imgs, mu, logvar = model(imgs)
        
                recon_loss = reconstruction_loss(imgs, recon_imgs, options['reconstruction_loss_distr'])
                total_kl_loss = kl_divergence(mu, logvar)
                loss = recon_loss + beta * total_kl_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if e == (epochs - 1):
                    chunkstart = i*batch_size
                    chunksize = mu.shape[0]
                    Z[chunkstart:chunkstart + chunksize] = mu.detach().numpy()
                    y[chunkstart:chunkstart + chunksize] = labels

                if i % interval == 0:
                    loss = loss/len(imgs) if reduction=='mean' else loss
                    print(f'epoch {e}/{epochs} [{i*len(imgs)}/{len(train_dataloader.dataset)} ({100.*i/len(train_dataloader):.2f}%)]'
                        f'\tloss: {loss.item():.4f}'
                        f'\tlr: {scheduler.get_last_lr()}')
            scheduler.step()
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'Z': Z,
            'y': y
            }, options['saved_model_path'] + options['model_identifier'])

    print('training Classifier')
    model.fit_classifier(Z, y)
    
    return model


def load_model(identifier, path, model, optimizer):
    try:
        checkpoint = torch.load(path + identifier)
    except FileNotFoundError:
        print('no model file found')
        return (False, None, None)
    print('model loaded from file')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    Z = checkpoint['Z']
    y = checkpoint['y']
    return (True, Z, y)



