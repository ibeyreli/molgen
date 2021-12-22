"""
Model Classes for Deep Molecule Generator

author: ilayda beyreli kokundu
date 03/11/2021
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, latent_dim=128, input_size=(1,128,128), feature_size=100, out_size=2 ):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.feature_size = feature_size
        self.out_size = out_size
        self.label_emb = nn.Embedding(self.out_size, self.out_size)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim + self.feature_size + self.out_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.input_size))),
            nn.Tanh()
        )

    def forward(self, features, noise, labels):
        # Concatenate label embedding, features and noise to produce input
        gen_input = torch.cat((self.label_emb(labels), features, noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), * img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, latent_dim=128, input_size=(1,128,128), out_size=2 ):

        super(Discriminator, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.out_size = out_size

        self.label_embedding = nn.Embedding(self.out_size, self.out_size)

        self.model = nn.Sequential(
            nn.Linear(self.out_size + int(np.prod(self.input_size)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, graph, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((graph.view(graph.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


