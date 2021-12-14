"""
Data Parser for Deep Molecule Generator

author: ilayda beyreli kokundu, mustafa duymus
date 03/11/2021
"""
import re
import os, sys
import pickle
import numpy as np
import pandas as pd

import networkx as nx
import pysmiles as ps

RAW_FILE_DIF = "E:\\BILKENT_CS\\CS585"

def molecule_counter(s, d=None):
    if d is None: d=dict()
    
    r = re.split(r'(\d+)|(?=[A-Z])', s)
    
    for i in range(len(r)):
        if r[i] is None or r[i] == '': continue
        
        if r[i].isdigit() : 
            continue
        else:
            if i == len(r)-1 or r[i+1] is None or r[i+1]=='' : # Take care of the single atoms
                c = 1
            else:
                c = int(r[i+1])
                
            if r[i] in d.keys():
                d[r[i]] = max(d[r[i]], c)
            else:
                d[r[i]] = c

    return d
     

raw_file = pd.read_csv(os.path.join(RAW_FILE_DIF,"data.csv"), sep=";")
# Example Reading:
#      ChEMBL ID              Name  ... Molecular Formula                                             Smiles
# 0  CHEMBL3989817  DIPROLEANDOMYCIN  ...        C41H69NO14  CCC(=O)O[C@H]1[C@H](C)O[C@@H](O[C@@H]2[C@@H](C...
# 1   CHEMBL306107               NaN  ...      C17H14Cl2N4O          COc1ccc(Cl)cc1-c1cc(Nc2ccc(Cl)cc2)nc(N)n1

input = raw_file.values[:,-2]
# INPUT  : (2105464,) ['C41H69NO14' 'C17H14Cl2N4O']

target = raw_file.values[:,-1]
# TARGET : (2105464,) ['CCC(=O)O[C@H]1[C@H](C)O[C@@H](O[C@@H]2[C@@H](C)C(=O)O[C@H](C)[C@H](C)[C@H](OC(=O)CC)[C@@H](C)C(=O)[C@]3(CO3)C[C@H](C)[C@H](O[C@@H]3O[C@H](C)C[C@H](N(C)C)[C@H]3O)[C@H]2C)C[C@@H]1OC'
#                      'COc1ccc(Cl)cc1-c1cc(Nc2ccc(Cl)cc2)nc(N)n1']

# Find the maximum possible number of each atom in the entire set
counts = dict()
#for molecule in input:
#    counts = molecule_counter(molecule, counts)
#
#with open("counts.pkl", "wb") as fout:
#    pickle. dump(counts, fout)

counts = pickle.load("counts.pkl")

n = max(counts.values())
features = np.zeros((target.shape[0], len(counts.keys())))
data = []
for (i, smiles) in enumerate(target):
    molecule = ps.read_smiles(smiles, explicit_hydrogen=True)
    A = nx.adjacency_matrix(molecule).todense()
    print(A)

