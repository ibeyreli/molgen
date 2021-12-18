"""
Utiliy Functions for Deep Molecule Generator

author: ilayda beyreli kokundu
date 03/11/2021
"""
import numpy as np
import os
import math

from torch.utils.data import DataLoader
from torch.autograd import Variable

import pdb
import time
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

from pysmiles import write_smiles
import logging
logging.getLogger('pysmiles').setLevel(logging.CRITICAL)  # Anything higher than warning


def sample_molecule(n_row, batches_done):
    """Saves a grid of generated molecules"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels
    save_sample(gen_imgs.data, "molecules/%d.png" % batches_done, nrow=n_row, normalize=True)

def plot_learing_curve(tgl, tdl, vgl, vdl):
    fig=plt.figure(figsize=(16,24))
    



