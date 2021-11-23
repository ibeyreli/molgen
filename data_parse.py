"""
Data PArser for Deep Molecule Generator

author: ilayda beyreli kokundu, mustafa duymus
date 03/11/2021
"""
import os, sys
import numpy as np
import pandas as pd

RAW_FILE_DIF = "E:\\BILKENT_CS\\CS585\\molgen"

raw_file = pd.read_csv(os.path.join(RAW_FILE_DIF,"data.csv"))
print(raw_file.head(2))