"""
Deep Molecule Generator

author: ilayda beyreli kokundu
date 03/11/2021
"""
import os
import json
import math

import pdb
import time
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

from pysmiles import write_smiles
import logging
logging.getLogger('pysmiles').setLevel(logging.CRITICAL)  # Anything higher than warning

from utils import *
from molgen import *

DATA_FILE = "/mnt/ilayda/molgen_data"
system_gpu_mask = "1"
batch_size = 32
max_epoch = 10
l_rate = 1e-4
latent_dim = 64
debug = False
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
#    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    print('Total:   ', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'GB')
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    
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
        return self.__data[i], self.__features[i]



files = os.listdir(DATA_FILE)
try:
    files.remove("features.npy")
    files.remove("data.csv")
except ValueError:
    print("features.npy or data.csv is not in the file list" )

features = np.load("/mnt/ilayda/molgen_data/features.npy")
dataset = np.array(sorted(files))

print("Dataset of shape:", dataset.shape, " Feature of shape:", features.shape)
features = features[:dataset.shape[0],:5]

x_train, x_test, f_train, f_test = train_test_split(dataset, features, train_size=8/10, shuffle=True)

train_set = Molecules(x_train, f_train)
test_set = Molecules(x_test, f_test)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last = True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

ref_vec = generate_reference()
ref_vec = torch.from_numpy(ref_vec[::9]).double().to(device)
f_size = features.shape[1]

sample = load_molecule(dataset[0])
print("Sample shape:",sample.shape)
# -----------------------------------------------
# Model Setup
# -----------------------------------------------
def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False

print("Model init...")
generator = Generator().to(device)
print("Generator initialized.")
discriminator = Discriminator().to(device)
print("Discriminator initialized.")
# print(discriminator)
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=l_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=l_rate)

print("Training...")
# -----------------------------------------------
# Training
# -----------------------------------------------
val_d_loss = [1000]
val_g_loss = [1000]

train_d_loss = []
train_g_loss = []

d_early_stop = 7
g_early_stop = 7

for epoch in range(max_epoch):
    for batch_idx, (molecules, feats ) in enumerate(train_loader):
        batch_d_loss = []
        batch_g_loss = []

        molecules = load_molecules(molecules)
        # molecules, feats = torch.from_numpy(molecules).to(device), torch.from_numpy(feats).to(device)
        molecules = torch.from_numpy(molecules).to(device)
        feats = feats.to(device)
        
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(molecules.type(FloatTensor))
        feats = Variable(feats.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        gen_feats = Variable(FloatTensor(np.random.randint(0, 160, size=(batch_size, feats.size(1) ) ) ) )
        
        # Generate a batch of images
        gen_imgs = generator(z,gen_feats)
        
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs)
        g_loss = F.mse_loss(validity, valid)
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TO DO:
        # Check if the martix is symmetric A-A^t
        # Then get row sum, and multiply with ref vector
        # Loss measures generator's ability to generate meaningful molecules
        sym_loss = torch.abs(symmetry_loss(gen_imgs)).to(device)
        b_loss = torch.abs(bond_loss(gen_imgs.double(), ref_vec)).to(device)
        # print(g_loss.dtype, b_loss.dtype, sym_loss.dtype)

        batch_g_loss.append(g_loss.item())

        g_loss = g_loss + sym_loss.mean() + b_loss.mean()
        g_loss.sum().backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs)
        d_real_loss = F.mse_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach())
        d_fake_loss = F.mse_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        batch_d_loss.append(d_loss.item())
        d_loss.backward()
        optimizer_D.step()


        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [S loss: %f] [B loss: %f]"
            % (epoch, max_epoch, batch_idx, len(train_loader), d_loss.item(),
               g_loss.mean().item(), sym_loss.mean().item(), b_loss.mean().item() )
        )
        if debug and batch_idx == 10: break

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

        for batch_idx, (molecules, feats) in enumerate(test_loader):

            molecules = load_molecules(molecules)
            # molecules, feats = torch.from_numpy(molecules).to(device), torch.from_numpy(feats).to(device)
            molecules = torch.from_numpy(molecules).to(device)
            feats = feats.to(device)

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_molecules = Variable(molecules.type(FloatTensor))
            feats = Variable(feats.type(LongTensor))

            # -----------------
            #  Generator
            # -----------------

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_feats = Variable(FloatTensor(np.random.randint(0, 160, size=(batch_size, feats.size(1) ) ) ) )

            # Generate a batch of images
            gen_imgs = generator(z, gen_feats)
            
            # Loss measures generator's ability to generate meaningful molecules
            sym_loss = torch.abs(symmetry_loss(gen_imgs)).to(device)
            b_loss = torch.abs(bond_loss(gen_imgs.double(), ref_vec)).to(device)
            
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs)
            g_loss = F.mse_loss(validity, valid) + b_loss.mean() + sym_loss.mean()
            g_losses.append(g_loss.item())

            # ---------------------
            # Discriminator
            # ---------------------

            # Loss for real images
            validity_real = discriminator(real_imgs)
            d_real_loss = F.mse_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach())
            d_fake_loss = F.mse_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_losses.append(d_loss.item())

            if debug and batch_idx == 10: break

        # print(d_losses, g_losses)
        if np.mean(d_losses) > val_d_loss[-1] and d_early_stop != 0:
            d_early_stop -= 1
        elif d_early_stop != 0:
            d_early_stop = 7
        val_d_loss.append(np.mean(d_losses))

        if np.mean(g_losses) > val_g_loss[-1] and g_early_stop != 0:
            g_early_stop -= 1
        elif g_early_stop != 0:
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

        batches_done = epoch * len(test_loader) + batch_idx
        if batches_done % 100 == 0:
           # sample_image(n_row=10, batches_done=batches_done)
           torch.save(generator.state_dict(), "generator_checkpoint"+str(batches_done)+".pth")
           torch.save(discriminator.state_dict(), "discriminator_checkpoint"+str(batches_done)+".pth")

fout = open("runreport.txt", "w+")
fout.write("Train Losses: Generative \n")
for l in train_g_loss:
    fout.write("%f\n" % l)
fout.write( "-"*20+"\n\n")
fout.write("Train Losses: Discriminative \n")
for l in train_d_loss:
    fout.write("%f\n" % l)
fout.write( "-"*20+"\n\n")
fout.write("Validation Losses: Generative \n")
for l in val_g_loss:
    fout.write("%f\n" % l)
fout.write( "-"*20+"\n\n")
fout.write("Validation Losses: Discriminative \n")
for l in val_d_loss:
    fout.write("%f\n" % l)

fout.close()
plot_learing_curve(train_g_loss, train_d_loss, val_g_loss, val_d_loss)
