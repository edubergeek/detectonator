"""
Implement a neural network model for Numerai data using a keras multi-layer perceptron.
To get started, install the required packages: pip install pandas numpy tensorflow keras
"""
# #!pip install -U tensorboard_plugin_profile

# +
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from KerasModel import KerasModel, PlotLoss, BestEpoch
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, CategoryEncoding
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping


# +
""" 
Derived from NumeraiKeras class to implement 2 layer MLP model
based on example_model.py
"""

class DetectonatorCNN(KerasModel):
  def __init__(self, hparam):
    super().__init__() 

    self.SetModelName("cnnmc")
    self.SetModelPath(self.GetModelName())
    self.SetModelVersion(1)
    self.SetModelTrial(1)
    self.xDim = (100,100,1)
    self.yDim = 5

    # Construct the model
    self.model = self.BuildModel(hparam)


  def BuildNetwork(self, inputs, dim=128, layers=3, expand=2, kernel=3, activation='relu', final_layers=1, final_units=4096, final_expansion=1, final_dropout=0.0, final_activation='relu'):

    # Construct the network
    act = activation
    pad = 'same'
    m = inputs
    for l in range(layers):
      m = Conv2D(filters=dim, kernel_size=kernel, activation=act)(m)
      m = MaxPooling2D(pool_size=2)(m)
      if self.dropout[l] > 0.0:
        m = Dropout(self.dropout[l])(m)
      dim *= expand
    m = Flatten()(m)

    units = final_units
    for l in range(final_layers):
      m = Dense(units, activation=final_activation)(m)
      if final_dropout > 0.0:
        m = Dropout(final_dropout)(m)
      units *= final_expansion
    m = Dense(self.yDim, activation='softmax')(m)

    return m

  def BuildModel(self, hparam):    
    # get the hyperparameters
    hp_activation = hparam['activation']
    hp_dropout = hparam['dropout']
    hp_learningRate = hparam['learning_rate']
    hp_units = hparam['units']
    hp_expansion = hparam['expansion']
    hp_kernel = hparam['kernel']
    hp_layers = hparam['layers']
    hp_finalUnits = hparam['final_units']
    hp_finalLayers = hparam['final_layers']
    hp_finalExpansion = hparam['final_expansion']
    hp_finalDropout = hparam['final_dropout']
    hp_finalActivation = hparam['final_activation']

    # initialize the input shape and channel dimension
    inputs = Input(shape=self.xDim)

    # initialize dropout for each layer
    self.dropout=np.full((hp_layers),hp_dropout)

    # Construct the network
    nn = self.BuildNetwork(inputs, dim=hp_units, layers=hp_layers, expand=hp_expansion, kernel=hp_kernel, activation=hp_activation, final_layers=hp_finalLayers, final_units=hp_finalUnits, final_expansion=hp_finalExpansion, final_dropout=hp_finalDropout, final_activation=hp_finalActivation)

    m = Model(inputs=inputs, outputs=nn, name=self.GetModelName())
    self.SetModel(m)
    self.SetLearningRate(hp_learningRate)
    self.SetBatchSize(512)
    self.SetOptimizer(Adam)
    self.SetLoss('categorical_crossentropy')
    self.SetMetrics(['mse','mae','accuracy'])
    self.Compile()
     
    return m


# -

def main():
#  hyperParam = {
#    'activation': 'tanh',
#    'dropout': 0.25,
#    'units': 96,
#    'expansion': 3,
#    'kernel': 3,
#    'layers': 1,
#    'final_units': 1024,
#    'final_layers': 2,
#    'final_expansion': 2,
#    'final_dropout': 0.25,
#    'final_activation': 'relu',
#    'learning_rate': 0.0003,
#    'batch_size': 512,
#  }
  hyperParam = {
    'activation': 'relu',
    'dropout': 0.0,
    'units': 64,
    'expansion': 2,
    'kernel': 3,
    'layers': 3,
    'final_units': 512,
    'final_layers': 2,
    'final_expansion': 1,
    'final_dropout': 0.0,
    'final_activation': 'relu',
    'learning_rate': 1e-3,
    'batch_size': 128,
  }

  dm = DetectonatorCNN(hyperParam)

  dm.SetModelRevision(1)
  dm.SetModelVersion(4)
  dm.SetModelTrial(1)
  dm.model.summary()
  
  print(dm.GetModelFile())
  dm.SaveModel()


if __name__ == '__main__':
  main()


