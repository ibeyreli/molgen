"""
Data Parser for Deep Molecule Generator

author: ilayda beyreli kokundu
date 03/11/2021
"""
import re, json
import os, sys

import numpy as np
import pandas as pd

import networkx as nx
import pysmiles as ps

import logging
logging.getLogger('pysmiles').setLevel(logging.CRITICAL) 

import warnings
warnings.filterwarnings("ignore")


#RAW_FILE_DIF = "E:\\BILKENT_CS\\CS585" # Windows
RAW_FILE_DIF = "/mnt/ilayda/molgen_data" # Neo - Ubuntu

def molecule_counter(s, d=None, e=False):
    if d is None:
        d=dict()
    try:
        r = re.split(r'(\d+)|(?=[A-Z])', s)
    except TypeError:
        # print("Problematic string!", s)
        e = True
        return d, e

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

    return d, e


raw_file = pd.read_csv(os.path.join(RAW_FILE_DIF,"data.csv"), sep=";")
# Example Reading:
#      ChEMBL ID              Name  ... Molecular Formula                                             Smiles
# 0  CHEMBL3989817  DIPROLEANDOMYCIN  ...        C41H69NO14  CCC(=O)O[C@H]1[C@H](C)O[C@@H](O[C@@H]2[C@@H](C...
# 1   CHEMBL306107               NaN  ...      C17H14Cl2N4O          COc1ccc(Cl)cc1-c1cc(Nc2ccc(Cl)cc2)nc(N)n1

input = raw_file.values[:,-2]
# INPUT  : (2105464,) ['C41H69NO14' 'C17H14Cl2N4O']
# print("INPUT:", input)
target = raw_file.values[:,-1]
# TARGET : (2105464,) ['CCC(=O)O[C@H]1[C@H](C)O[C@@H](O[C@@H]2[C@@H](C)C(=O)O[C@H](C)[C@H](C)[C@H](OC(=O)CC)[C@@H](C)C(=O)[C@]3(CO3)C[C@H](C)[C@H](O[C@@H]3O[C@H](C)C[C@H](N(C)C)[C@H]3O)[C@H]2C)C[C@@H]1OC'
#                      'COc1ccc(Cl)cc1-c1cc(Nc2ccc(Cl)cc2)nc(N)n1']
# print("TARGET:", target.shape)
# Find the maximum possible number of each atom in the entire set

counts = dict()

#for molecule in input:
#    counts, _  = molecule_counter(molecule, counts)

l = [] # the list for ok molecules
"""
for (i,molecule) in enumerate(input):
    counts, e = molecule_counter(molecule, counts)
    if not e: l.append(True)
    else: l.append(False)
    if i % 100000 == 0:
        print("At molecule", i)

with open("counts.json", "w") as fout:
    json.dump(counts, fout)
with open("indices.json", "w") as fout:
    json.dump(l, fout)
"""
with open("counts.json", "r") as fin:
    counts = json.load(fin)
with open("indices.json", "r") as fin:
    l = json.load(fin)

# print("Data set of ", len(l), "molecules")
# print("Dictionary", counts)
order = np.array( [ [key, int(counts[key]) ] for key in list(counts.keys()) ], dtype=object )
# print("The Order!", order[0:2,1].astype(int))

# Drop problematic samples
input = input[l]
target = target[l]

fout = open("data_parse_log.txt", "w")

n = sum(counts.values())
set_size = 30000
features = np.zeros((set_size, len(counts.keys()) )).astype(dtype=np.uint8)

#data = np.zeros((set_size, n, n), dtype=np.bool_)
badlines = []
k = 0
for (i, smiles) in enumerate(target):
    try:
        molecule = ps.read_smiles(smiles, explicit_hydrogen=True)
    except:
        fout.write("Error at line: %d \n" % i)
        badlines.append(i)
#        l.pop(i)
        continue
    A = nx.adjacency_matrix(molecule).todense().astype(np.bool_)
    #print("A and Nodes \n", A, molecule.nodes(data='element'))
    atoms, _ = molecule_counter(input[i])
    # get a new matrix to fill
    Aprime = np.zeros((n, n), dtype=np.bool_)    
    for key in atoms.keys():
        j = list(counts.keys()).index(key)
        # set the feature
        features[i,j] = atoms[key]
        # find the start index 
        start = sum(order[:j,1].copy().astype(int))
        # print("Debugging here!", key,j,start)
        nodes = [ind for (ind,element) in enumerate(molecule.nodes(data='element')) if element[-1] == key ]
        nodes2 = [ind+start for (ind,element) in enumerate(molecule.nodes(data='element')) if element[-1] == key ]
        # print(len(nodes))
        Aprime[nodes2, nodes2] = A[nodes,nodes].copy()
    np.save(os.path.join(RAW_FILE_DIF,"adjacency_"+str(k)+".npy"), Aprime)
#    data[i, : ,: ] = Aprime.copy()
    k += 1
    if i % 1000 == 0: print("At line:", i)
    if i == set_size-1: break

print("Total samples:", k)

# Not used due to memory issues!
#data = np.delete(np.array(data), badlines, axis=0)
#np.save(os.path.join(RAW_FILE_DIF,"data.npy"), data, allow_pickle=True)

features = np.delete(features, badlines, axis=0)
np.save(os.path.join(RAW_FILE_DIF,"features.npy"), features, allow_pickle=True)

print("Final shape", features.shape)
