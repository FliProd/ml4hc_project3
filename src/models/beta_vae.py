# Based on: https://github.com/1Konny/Beta-VAE/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import umap
import umap.plot


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        if tensor.shape[0] != self.size:
            # corner case for last batch
            tensor = tensor.view((tensor.shape[0], *self.size[1:]))
        else:
            tensor = tensor.view(self.size)
        return tensor


class BetaVAE_CLF(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, batch_size=32, nc=1, classifier='SVM', classifier_options={}):
        super(BetaVAE_CLF, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 64, 64
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 64, 64
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  32,  32
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  16,  16
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 2, 1),         # B, 256,  4,  4
            nn.ReLU(True),
            View((batch_size, 4096)),                    # B, 256
            nn.Linear(4096, z_dim*2),            # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 4096),               # B, 256
            View((batch_size, 256, 4, 4)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4, 2, 1),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 128, 128
        )
        self.weight_init()

        # initialize classifier which will predict on latent space
        if classifier == 'SVM':
            self.classifier = SVC(**classifier_options)
        elif classifier == 'RFS':
            self.classifier = RandomForestClassifier(**classifier_options)
        elif classifier == 'KNN':
            self.classifier = KNeighborsClassifier(**classifier_options)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar) 
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def plot_latent_space(self, Z, y):
        reducer = umap.UMAP()
        reducer.fit(Z)
        umap.plot.points(reducer, labels=y)
        umap.plot.plt.savefig('reports/figures/latent_space_umap.png')


    def fit_classifier(self, Z, y):
        self.classifier.fit(Z, y)

    def predict(self, X):
        print('predicting')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        Z = self._encode(X)[:,:self.z_dim].detach().numpy()
        return self.classifier.predict(Z)



def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)