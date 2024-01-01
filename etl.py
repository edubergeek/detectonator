# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# #!pip install pyarrow
# #!pip install sklearn scikit-image
# !pip install astropy

# +
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import json
import argparse
#import csv
import tensorflow as tf
import skimage.io as skio

from enum import IntEnum
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime
from NEOETL import _floatvector_feature, _float_feature, _int64_feature, _bytes_feature, _dtype_feature, TrainingSet, ETL, NEOETL

# +
IS_JUPYTER = True

if IS_JUPYTER:
  #sys.argv.append('--train')
  #sys.argv.append('--balance')
  sys.argv.append('--split')
  sys.argv.append('0.9')
  sys.argv.append('--valsplit')
  sys.argv.append('0.05')
  sys.argv.append('--shard')
  sys.argv.append('1024')


# +


parser = argparse.ArgumentParser()
#parser.add_argument("--train", action='store_true', help="ETL training set")
#parser.add_argument("--test", action='store_true', help="ETL live data")
#parser.add_argument("--balance", action='store_true', help="balance target values by class")
parser.add_argument("--shard", type=int, default=1024, help="shard size")
parser.add_argument("--split", type=float, default=0.9, help="train split percentage")
parser.add_argument("--valsplit", type=float, default=0.05, help="valid split percentage")




# +
print(sys.argv)
if IS_JUPYTER:
  args = parser.parse_args(sys.argv[3:])
else:
  args = parser.parse_args()

print(args)

# -

dataRoot = '../data'
if not os.path.exists(dataRoot):
  os.makedirs(dataRoot)

# !rm -f ../data/*

# +
train_split = args.split
valid_split = args.valsplit
test_split = 1.0 - (train_split + valid_split)

print('Train/Valid/Test = %2.1f%%/%2.1f%%/%2.1f%%' %(100*train_split, 100*valid_split, 100*test_split))
# -


etl = NEOETL('../FITStamp', dataRoot, train_split = train_split, valid_split = valid_split, shard_size = args.shard, manifest='manifest.csv')

etl.OpenDatasets()

# +
# Load training examples with train/valid split
print("Loading training dataset ...")
etl.Load(TrainingSet.TRAIN)
print("Sharding %d training examples ..." % (etl.examples[TrainingSet.TRAIN]))
etl.SaveDataset(TrainingSet.TRAIN, TrainingSet.TRAIN)
# Load validation examples with train/valid split

etl.Load(TrainingSet.VALID, reload=False)
print("Sharding %d validation examples ..." % (etl.examples[TrainingSet.VALID]))
etl.SaveDataset(TrainingSet.VALID, TrainingSet.VALID)
# -

etl.Load(TrainingSet.TEST, reload=False)
print("Sharding %d test examples ..." % (etl.examples[TrainingSet.TEST]))
etl.SaveDataset(TrainingSet.TEST, TrainingSet.TEST)

etl.CloseDatasets()
print("Finished")



