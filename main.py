"""
Deep Molecule Generator

author: ilayda beyreli kokundu
date 03/11/2021
"""
import os
import math

from torch.utils.data import DataLoader
from torch.autograd import Variable

import pdb
import time
import torch

import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

from pysmiles import write_smiles
import logging
logging.getLogger('pysmiles').setLevel(logging.CRITICAL)  # Anything higher than warning

DATA_FILE = "/mnt/ilayda/mogen_data"
system_gpu_mask = "0"
batch_size = 64
max_epoch = 1000
l_rate = 1e-4
latent_dim = 128

# Seeds
torch.manual_seed(42)
np.random.seed(42)

from molgen import *

# Setting device on GPU if available, else CPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= system_gpu_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    print('Total:   ', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'GB')

# -----------------------------------------------
# Dataset Preperation
# -----------------------------------------------

class Molecules(Dataset):
    def __init__(self, data, features, **kwargs):
        self.__data = data
        self.__features = features

    def __len__(self):
        return self.__features.shape[0]

    def __getitem__(self, i):
        return self.__data[i,:], self.__features[i,:]



files = os.listdir(DATA_FILE)
files.delete("features.npy")
features = np.load("/mnt/ilayda/molgen_data/features.npy")
dataset = []
for file in files:
    dataset.append(np.memmap(os.path.join(DATA_FILE,file), mode="r"))

x_train, x_test, f_train, f_test = train_test_split(dataset, labels, train_size=8/10, shuffle=True)

train_set = Molecules(x_train, f_train)
test_set = Molecules(x_test, f_test)

train_loader = Dataloader(train_set, batch_size=batch_size, shuffle=True)
test_loader = Dataloader(test_set, batch_size=batch_size, shuffle=False)

# -----------------------------------------------
# Model Setup
# -----------------------------------------------
def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False

generator = Generator().to(device)
discriminator = Dscriminator().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=l_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=l_rate)

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_avaliable() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_avaliable() else torch.LongTensor

# -----------------------------------------------
# Training
# -----------------------------------------------
val_d_loss = []
val_g_loss = []

train_d_loss = []
train_g_loss = []

d_early_stop = 7
g_early_stop = 7

for epoch in range(max_epoch):
    for batch-idx, (molecules, feats) in enumerate(traiin_loader):
        batch_d_loss = []
        batch_g_loss = []
        molecules, feats = molecules.to(device), feats.to(device)
        
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_molecules = Variable(molecules.type(FloatTensor))
        #feats = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = F.MSEloss(validity, valid)
        batch_g_loss.append(g_loss.item())
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = F.MSEloss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = F.MSEloss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        batch_d_loss.append(d_loss.item())
        d_loss.backward()
        optimizer_D.step()


        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, max_epoch, i, len(dataloader), d_loss.item(), g_loss.item())
        )
    train_d_loss.append(np.mean(batch_d_loss))
    train_g_loss.append(np.mean(batch_g_loss))
    #--------------------
    # Validation
    #--------------------
    with torch.no_grad():
        generator.eval()
        discriminator.eval()
        d_losses = []
        g_losses = []

        for batch-idx, (molecules, feats) in enumerate(test_loader):

            molecules, feats = molecules.to(device), feats.to(device)

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_molecules = Variable(molecules.type(FloatTensor))
            #feats = Variable(labels.type(LongTensor))

            # -----------------
            #  Generator
            # -----------------

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = F.MSEloss(validity, valid)
            g_losses.append(g_loss.item())

            # ---------------------
            # Discriminator
            # ---------------------

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = F.MSEloss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = F.MSEloss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

        if np.mean(d_losses) > val_d_loss[-1]:
            d_early_stop -= 1
        else:
            d_early_stop = 7
        val_d_loss.append(np.mean(d_losses))

        if np.mean(g_losses) > val_g_loss[-1]:
            g_early_stop -= 1
        else:
            g_early_stop = 7
        val_g_loss.append(np.mean(g_losses))

        if d_early_stop == 0 and g_early_stop == 0:
             discriminator.apply(freeze_layer)
             generator.apply(freeze_layer)
             torch.save(generator.state_dict(), "generator_best.pth")    
             torch.save(discriminator.state_dict(), "discriminator_best.pth")    
             print("Training Done!")
             break

        elif d_early_stop == 0:
             discriminator.apply(freeze_layer)
             print("Discriminator Freezed!")
	elif g_early_stop == 0:
             generator.apply(freeze_layer)
             print("Generator Freezed!")

        batches_done = epoch * len(dataloader) + i
        if batches_done % 100 == 0:
            # sample_image(n_row=10, batches_done=batches_done)
            torch.save(generator.state_dict(), "generator_checkpoint"+str(batches_done)+".pth")    
            torch.save(discriminator.state_dict(), "discriminator_checkpoint"+str(batches_done)+".pth")    

fout = open("runreport.txt", "w+")
fout.write("Train Losses: Generative \n")

for l in train_g_loss:
    fout.write("%f\n" % l)
    
fout.write("Train Losses: Discriminative \n")
for l in train_d_loss:
    fout.write("%f\n" % l)

fout.write("Validation Losses: Generative \n")
for l in val_g_loss:
    fout.write("%f\n" % l)
    
fout.write("Validation Losses: Discriminative \n")
for l in val_d_loss:
    fout.write("%f\n" % l)

fout.close()
plot_learing_curve(train_g_loss, train_d_loss, val_g_loss, val_d_loss)
