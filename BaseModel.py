#!/usr/bin/env python
"""
Base keras model class
"""

import os

""" 
Keras model base class
"""
class BaseModel:
  def __init__(self):
    self.modelVersion = 1
    self.modelRevision = 0
    self.modelTrial = 1
    self.modelEpoch = 0
    self.sModelSuffix = ""
    self.sModelPath = "."
    self.isTraining = False

  def SetModelPath(self, path):
    self.sModelPath = path
    if not os.path.exists(path):
      os.makedirs(path)

  def GetModelPath(self):
    return self.sModelPath

  def SetModelName(self, name):
    self.sModelName = name

  def SetModelVersion(self, version):
    self.modelVersion = version

  def GetModelVersion(self):
    return self.modelVersion

  def SetModelRevision(self, revision):
    self.modelRevision = revision

  def GetModelRevision(self):
    return self.modelRevision

  def SetModelTrial(self, trial):
    self.modelTrial = trial

  def GetModelTrial(self):
    return self.modelTrial

  def SetModelEpoch(self, epoch):
    self.modelEpoch = epoch

  def SetModelSuffix(self, suffix):
    self.sModelSuffix = suffix

  def GetModelName(self):
    return self.sModelName

  def GetModelFullName(self):
    return "%sv%dr%d" %(self.sModelName, self.modelVersion, self.modelRevision)

  def GetModelFile(self):
    return "%sv%dr%dt%d-e%d%s" %(os.path.join(self.sModelPath, self.sModelName), self.modelVersion, self.modelRevision, self.modelTrial, self.modelEpoch, self.sModelSuffix)

