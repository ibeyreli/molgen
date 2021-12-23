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
    def __init__(self, latent_dim=64, input_size=(1,674,674), out_size=5):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.out_size = out_size
        self.label_emb = nn.Embedding(self.out_size, self.out_size)

        self.model = nn.Sequential(
                     nn.Linear(self.latent_dim + self.out_size, 128),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Linear(128, 512),
                     nn.BatchNorm1d(512),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Linear(512, int(np.prod(self.input_size)) ),
                     nn.Tanh(),
                     )

    def forward(self, noise, labels):
        # Concatenate label embedding, features and noise to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.input_size)
        return img


class Discriminator(nn.Module):
    def __init__(self, latent_dim=64, input_size=(1,674,674), out_size=5 ):

        super(Discriminator, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.out_size = out_size

        self.label_embedding = nn.Embedding(self.out_size, self.out_size)

        self.model = nn.Sequential(
            nn.Linear(self.out_size + int(np.prod(self.input_size)), 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, graph, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((graph.view(graph.size(0), -1), 
                          labels.float().view(labels.size(0), -1)
# self.label_embedding( torch.clamp(labels, min=0.0, max=1.0) ) 
                        ), -1)
        validity = self.model(d_in)
        return validity


