"""
Utiliy Functions for Deep Molecule Generator

author: ilayda beyreli kokundu
date 03/11/2021
"""
import os, json
import math

import pdb
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

from pysmiles import write_smiles
import logging
logging.getLogger('pysmiles').setLevel(logging.CRITICAL)  # Anything higher than warning


#def sample_molecule(n_row, batches_done):
#    """Saves a grid of generated molecules"""
#   # Sample noise
#    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
#    # Get labels ranging from 0 to n_classes for n rows
#    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
#    labels = Variable(LongTensor(labels))
#    gen_imgs = generator(z, labels
#    save_sample(gen_imgs.data, "molecules/%d.png" % batches_done, nrow=n_row, normalize=True)

def plot_learing_curve(tgl, tdl, vgl, vdl):
    fig, ax = plt.subplots(figsize=(16, 20))
    
    ax.plot(tgl,  color='green', label='Train Generator')
    ax.plot(tgd,  color='red', label='Train Discriminator')
    ax.plot(vgl,  color='blue', label='Validation Generator')
    ax.plot(vgd,  color='orange', label='Validation Discriminator')
    
    ax.set_title("Learning Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    
    plt.legend()
    fig.savefig("between_scaled.png")

def generate_reference(file_name="counts.json"):
    counts = dict()
    with open(file_name, "r") as fin:
        counts = json.load(fin)
    n = sum(counts.values())
    ref_vec = np.zeros((n,1))# .fill(0)
    i = 0
    for key in counts.keys():
        if key == 'C':
            ref_vec[i:counts[key]] = 0.25
        if key == 'H':
            ref_vec[i:counts[key]] = 1.0
        if key == '0':
            ref_vec[i:counts[key]] = 0.50
        if key == 'N':
            ref_vec[i:counts[key]] = 0.33
        i += counts[key]
    return ref_vec

def symmetry_loss(output): # custom loss function
    loss = output - torch.transpose(output, 0, 1) #beware of dim0 and dim1
    return loss.float() #torch.tensor(loss, dtype=torch.float)

    
def bond_loss(output, ref_vector):
    loss = torch.sum( torch.mul( torch.sum(output, 1), ref_vector) )
    return loss.float() #torch.tensor(loss, dtype=torch.long)

def load_molecules(molecules):
    DATA_FILE = "/mnt/ilayda/molgen_data"
    dataset = []
    for file in sorted(molecules):
        #temp =  np.moveaxis(np.load( os.path.join(DATA_FILE,file) ), [0,1,2], [2,1,0])
        dataset.append( np.moveaxis(np.load( os.path.join(DATA_FILE,file) )[::9,::9].reshape((674,674,1)), [0,1,2], [2,1,0]) )
    dataset = np.asarray(dataset)
    return dataset

def load_molecule(molecule):
    DATA_FILE = "/mnt/ilayda/molgen_data"
    return np.moveaxis(np.load( os.path.join(DATA_FILE,molecule) )[::9,::9].reshape((674,674,1)), [0,1,2], [2,1,0])

