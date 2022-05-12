import torch
import torch.nn.functional as F

# Creating a PyTorch class
class MemoryEncoder(torch.nn.Module):
    def __init__(self, dimensionality):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(dimensionality, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, dimensionality),
            torch.nn.Sigmoid(),
        )

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(dimensionality, 1), torch.nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, h):
        return self.decoder(h)

    def forward(self, x):
        h = self.encode(x)
        z = self.decode(h)
        return h, z


class MemoryVAE(torch.nn.Module):
    def __init__(self, dimensionality):
        super().__init__()

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

        self.enc1 = torch.nn.Linear(dimensionality, 128)
        self.enc2 = torch.nn.Linear(128, 64)
        self.enc3 = torch.nn.Linear(128, 64)

        self.dec1 = torch.nn.Linear(64, 128)
        self.dec2 = torch.nn.Linear(128, dimensionality)

    def encode(self, x):
        x = F.relu(self.enc1(x))
        mu = self.enc2(x)
        sigma = torch.exp(self.enc3(x))

        h = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return h

    def decode(self, h):
        z = F.relu(self.dec1(h))
        z = torch.sigmoid(self.dec2(z))
        return z

    def forward(self, x):
        h = self.encode(x)
        z = self.decode(h)
        return h, z
