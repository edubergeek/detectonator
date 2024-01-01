#!/usr/bin/env python
# coding: utf-8
# %%
# !pip install astropy

# %%
import sys
import os
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score,confusion_matrix,accuracy_score,classification_report,roc_curve,auc
from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

#from xgboost import XGBRegressor
#import lightgbm as lgb
#from lightgbm import LGBMRegressor

# %%
from NEOETL import _floatvector_feature, _float_feature, _int64_feature, _bytes_feature, _dtype_feature, TrainingSet, ETL, NEOETL, load_fits
from KerasModel import KerasModel

# %%
TRANSFORM = 'transform'
def decode_tfr(record_bytes):
    schema =  {
      "id": tf.io.FixedLenFeature([], dtype=tf.string),
      "x":  tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing = True),
      "y":  tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing = True),
    }    
    example = tf.io.parse_single_example(record_bytes, schema)
    return example

def reshape_C6(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'C6':
        # clone the last 50 values 4 times 50 50 50 50
        temp1 = tf.slice(features['x'], [0], [1000])
        temp2 = tf.slice(features['x'], [1000], [50])
        temp3 = tf.repeat(temp2, 4)
        temp4 = tf.concat([temp1, temp3], 0)
        features['x'] = temp4
        yDim = int(t['arg1'])
        xDim = int(t['arg2'])
        features['x'] = tf.reshape(features['x'], [6, yDim, xDim])
        features['x'] = tf.transpose(features['x'], perm=[1, 2, 0])        
    return features

def reshape_YXZ(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'YXZ':
        yDim = int(t['arg1'])
        xDim = int(t['arg2'])
        zDim = int(t['arg3'])        
        features['x'] = tf.reshape(features['x'], [yDim, xDim, zDim])
    return features

def reshape_XY(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'XY':
        yDim = int(t['arg1'])
        xDim = int(t['arg2'])
        features['x'] = tf.reshape(features['x'], [xDim, yDim])        
        features['x'] = tf.transpose(features['x'])
    return features

def reshape_YX(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'YX':
        yDim = int(t['arg1'])
        xDim = int(t['arg2'])
        features['x'] = tf.reshape(features['x'], [yDim, xDim])
    return features

def reshape_Slice(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'Slice':
        fromDim = int(t['arg1'])
        toDim = int(t['arg2'])
        features['x'] = tf.slice(features['x'], [fromDim], [toDim])
    return features

def reshape_Even(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'Even':
        fromDim = int(t['arg1'])
        toDim = int(t['arg2'])
        features['x'] = tf.gather(features['x'], tf.constant(np.arange(fromDim, toDim*2, 2)))
        features['x'] = tf.slice(features['x'], [fromDim], [toDim])
    return features

def replace_NaN(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'NaN':
        featureVal = float(t['arg1'])
        mask = tf.math.is_nan(features['x'])
        maskVal = tf.cast(tf.ones_like(mask), tf.float32) * tf.constant(featureVal, dtype=tf.float32)
        features['x'] = tf.where(mask, maskVal, features['x'])
    return features

def replace_OneHot(features, targets):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'OneHot':
        nClass = int(t['arg1'])
        targets = tf.one_hot(targets, nClass)
        targets = tf.reshape(targets, [-1])
        #targets = tf.keras.utils.to_categorical(targets, num_classes=nClass)
    return features, targets

def reshape_Pad(features):
    for t in hyperParam[TRANSFORM]:
      if t['name'] == 'Pad':
        fromPos = int(t['arg1'])
        toPos = int(t['arg2'])
        featureVal = float(t['arg3'])
        # initialize an array of "featureVal" values in the correct shape
        #val = np.ones((toPos-fromPos)) * featureVal
        #pad = tf.constant(val, dtype=tf.float32)
        pad = tf.slice(features['x'], [0], [toPos-fromPos])
        features['x'] = tf.concat([features['x'], pad], 0)
        
    return features

# %%
def sample(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# %%
class MyModel(KerasModel):
  def __init__(self, modelname, modelpath, hyperParam):
    #self.sModelPath = "../round/%d" %(hyperParam["round"])
    self.sModelPath = modelpath
    self.sModelName = modelname
    self.modelVersion = hyperParam["version"]
    self.modelRevision = hyperParam["revision"]
    self.modelTrial = hyperParam["trial"]
    self.modelEpoch = hyperParam["epoch"]
    self.sModelSuffix = ""    
    self.batchSize = hyperParam["batch_size"]
    #self.dft = None

    self.hparam = hyperParam

#  def LoadModel(self):
#    self.modelFile = self.GetModelFile()
#    print("Loading ", self.modelFile)
#    #if self.hparam['arch'] == 'XGB':
#    #  self.model = XGBRegressor(max_depth=8, learning_rate=7e-3, n_estimators=6000, n_jobs=18, colsample_bytree=0.1)
#    #  self.model.load_model(self.modelFile)
#    #elif self.hparam['arch'] == 'LGB':
#    #  self.model = lgb.Booster(model_file=self.modelFile)
#    #else:
#    #  self.model = tf.keras.models.load_model(self.modelFile)
#    self.model = tf.keras.models.load_model(self.modelFile)

#  def GetModelFile(self):
#    return "%s/%sv%dr%dt%d-e%d%s" %(self.sModelPath, self.sModelName, self.modelVersion, self.modelRevision, self.modelTrial, self.modelEpoch, self.sModelSuffix)

#  def GetBestEpochFile(self):
#    return "%s/%s-r%d-best.csv" %(self.sModelPath, self.sModelName, self.modelRevision)

#  def LoadBestEpochs(self, predict='-'):
#    self.bestEpochFile = self.GetBestEpochFile()
#    print('BestEpochFile = ', self.bestEpochFile)
#    if os.path.exists(self.bestEpochFile):
#      self.best = np.loadtxt(self.bestEpochFile, delimiter=',', dtype='int')
#    else:
#      self.bestEpochFile = "%s/%s-r%d-best.csv" %(self.sModelPath, predict, self.modelRevision)
#      if os.path.exists(self.bestEpochFile):
#        self.best = np.loadtxt(self.bestEpochFile, delimiter=',', dtype='int')
#      else:
#        self.best = np.ones((41))
#    print(self.best)

#  def GetTrialModelFile(self, model, version, trial, epoch, predict):
#    if predict == '-':
#      return "%s/%sv%dr%dt%d-e%d%s" %(self.sModelPath, model, version, self.modelRevision, trial, epoch, self.sModelSuffix)
#    else:
#      return "%s/%s/%sv%dr%dt%d-e%d%s" %(self.sModelPath, predict, model, version, self.modelRevision, trial, epoch, self.sModelSuffix)

#  def GetModelFullName(self):
#    return "%sv%dr%d" %(self.sModelName, self.modelVersion, self.modelRevision)

#  def GetModelName(self):
#    return self.sModelName

  def GetList(self, f):
    return f.numpy().tolist()

  def GetDataSet(self, filenames, transform):
    at = AUTOTUNE
    
    dataset = (
      tf.data.TFRecordDataset(filenames, num_parallel_reads=at)
      .map(decode_tfr, num_parallel_calls=at)
    )
    
    if not transform == '-':
      for t in self.hparam[transform]:
        if t['name'] == 'C6':
          dataset = dataset.map(reshape_C6, num_parallel_calls=at)
        if t['name'] == 'XY':
          dataset = dataset.map(reshape_XY, num_parallel_calls=at)
        if t['name'] == 'YX':
          dataset = dataset.map(reshape_YX, num_parallel_calls=at)
        if t['name'] == 'YXZ':
          dataset = dataset.map(reshape_YXZ, num_parallel_calls=at)
        if t['name'] == 'Even':
          dataset = dataset.map(reshape_Even, num_parallel_calls=at)
        if t['name'] == 'Slice':
          dataset = dataset.map(reshape_Slice, num_parallel_calls=at)
        if t['name'] == 'NaN':
          dataset = dataset.map(replace_NaN, num_parallel_calls=at)
        if t['name'] == 'Pad':
          dataset = dataset.map(reshape_Pad, num_parallel_calls=at)
    
    dataset = dataset.batch(self.batchSize).prefetch(at).repeat(count=1)

    return dataset
       
  def SaveSubmissionCSV(self):
    # Save predictions as a CSV and upload to https://numer.ai
    submissionCSV = self.GetModelFullName() + ".csv"
    #print(self.dft['ypred'].shape)
    #print(self.dft['prediction'][0:5])
    #if (self.hparam['arch'] == 'AE' or self.hparam['arch'] == 'NR' or self.hparam['arch'] == 'NC') and self.dft['prediction'].ndim == 1:
    #  pred = [x for row in self.dft['prediction'] for x in row]
    #  #pred = self.dft['prediction'].to_numpy()
    #  self.dft['prediction'] = pred
    #print(self.dft['prediction'][0:5])
    self.dft.to_csv(submissionCSV, header=True, index=False)
    return submissionCSV

  def Predict(self, ds, transform):
    # Generate predictions
    b=0
    lid = []
    lpred = []
    lypred = []
    lyhat = []
    names = ["id", "y", "yhat", "predictions"]

    for features in ds:
      yId = nm.GetList(features['id'])
      yId = list(yId)
      yHat = nm.GetList(features['y'])
      yHat = list(yHat)
      if self.hparam['arch'] == 'AE':
        _, yPred = self.model.predict(features['x'], self.batchSize)
      #elif self.hparam['arch'] == 'XGB' or self.hparam['arch'] == 'LGB':
      #  yPred = self.model.predict(features['x'].numpy())
      elif self.hparam['arch'] == 'MC':
        pred = self.model.predict(features['x'], self.batchSize)
        yPred = np.argmax(pred, axis=1)
      else:
        yPred = self.model.predict(features['x'], self.batchSize)[:,0]
          #yPred = nm.model.predict(features['x'], self.batchSize)
      #for t in self.hparam[transform]:
      #  if t['name'] == 'Sparse':
      #    print(yPred.shape)
      #    yPred = self.PredictTargetX(yPred)
        
      lid.append(yId)
      if self.hparam['arch'] == 'MC':
        lpred.append(pred)
      lypred.append(yPred)
      lyhat.append(yHat)
      #batches.append(pa.RecordBatch.from_arrays([pa.array(yId),pa.array(yPred[:,0])], names=names))
      b+=1
      if not b % PROGRESS_INTERVAL:
        print(".", end="")#print(b, yId[0], yPred[:,0][0])#, features['x'][0][0:10])

    #self.prediction = pa.Table.from_arrays([pa.StringArray(lid), pa.array(lpred)], names=names)
    #self.prediction = pa.Table.from_batches(batches)
    print("done")
    self.label = [item.decode("utf-8") for sublist in lid for item in sublist] 
    self.dft = pd.DataFrame(self.label, columns = ['id'])
    if self.hparam['arch'] == 'MC':
      self.prediction = [item for sublist in lpred for item in sublist] 
      self.dft['prediction'] = self.prediction    
    self.yhat = [item[0] for sublist in lyhat for item in sublist] 
    self.dft['yhat'] = self.yhat    
    self.ypred = np.asarray([item for sublist in lypred for item in sublist])
    self.dft['ypred'] = self.ypred    

# %%
AUTOTUNE = tf.data.AUTOTUNE
PROGRESS_INTERVAL = 10


# %%
IS_JUPYTER = True
if IS_JUPYTER:
  sys.argv.append('--model')
  sys.argv.append('cnnmc')
  sys.argv.append('--arch')
  sys.argv.append('MC')
  sys.argv.append('--version')
  sys.argv.append('4')
  sys.argv.append('--revision')
  sys.argv.append('1')
  sys.argv.append('--trial')
  sys.argv.append('1')
  sys.argv.append('--epoch')
  sys.argv.append('17')
  sys.argv.append('--batch_size')
  sys.argv.append('256')
  sys.argv.append('--transform')
  sys.argv.append('YX,100,100|OneHot,5,0')


# %%
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model name")
parser.add_argument("--path", default="xyzzy", help="model path")
parser.add_argument("--version", type=int, help="model version")
parser.add_argument("--revision", type=int, help="model revision")
parser.add_argument("--trial", type=int, help="model trial")
parser.add_argument("--epoch", type=int, help="model epoch")
parser.add_argument("--transform", default='-', help="transform")
parser.add_argument("--arch", default="TF", help="model architecture")
parser.add_argument("--datadir", default="../data", help="data directory")
parser.add_argument("--filepat", default="test_*.tfr", help="prediction input file pattern")
parser.add_argument("--batch_size", type=int, default=8192, help="batch size")

if IS_JUPYTER:
  args = parser.parse_args(sys.argv[3:])
else:
  args = parser.parse_args()

print(args)


# %%
hyperParam = {
  'batch_size': args.batch_size,
}

hyperParam['model'] = args.model
hyperParam['version'] = args.version
hyperParam['revision'] = args.revision
hyperParam['trial'] = args.trial
hyperParam['epoch'] = args.epoch
hyperParam['arch'] = args.arch
hyperParam['transform'] = []
if not args.transform == '-':
  filters = args.transform.split('|')
  for f in range(len(filters)):
    param = filters[f].split(',')
    if len(param) == 3:
      hyperParam['transform'].append({
        'name': param[0],
        'arg1': param[1],
        'arg2': param[2]
        })
    if len(param) == 4:
      hyperParam['transform'].append({
        'name': param[0],
        'arg1': param[1],
        'arg2': param[2],
        'arg3': param[3]
        })

print(hyperParam)


# %%
# Create a MirroredStrategy.
#strategy = tf.distribute.MirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
#with strategy.scope():
  # Everything that creates variables should be under the strategy scope.
  # In general this is only model construction & `compile()`.
if args.path == 'xyzzy':
  args.path = args.model
nm = MyModel(args.model, args.path, hyperParam)


# %%
file_pattern = os.path.join(args.datadir, args.filepat)
test_filenames = tf.io.gfile.glob(file_pattern)
TRANSFORM='transform'
ds = nm.GetDataSet(test_filenames, TRANSFORM)


# %%
modelFile = nm.GetModelFile()
print("Loading model %s ..."%(modelFile))
#if hyperParam['arch'] == 'XGB':
#  nm.model = XGBRegressor(max_depth=8, learning_rate=7e-3, n_estimators=6000, n_jobs=18, colsample_bytree=0.1)
#  nm.model.load_model(modelFile)
#elif hyperParam['arch'] == 'LGB':
#  nm.model = lgb.Booster(model_file=modelFile)
#else:
#  nm.model = tf.keras.models.load_model(modelFile, custom_objects={'z': sample})
nm.model = tf.keras.models.load_model(modelFile, custom_objects={'z': sample})

# %%
if hyperParam['arch'] == 'NR' or hyperParam['arch'] == 'MC' or hyperParam['arch'] == 'AE':
  nm.model.summary()

# %%
nm.Predict(ds, TRANSFORM)

# %%
# Save the results for submission to Numerai
submission_file = nm.SaveSubmissionCSV()
print(submission_file)



# %%
nm.dft.head()

# %%
nm.dft['prediction'][0].shape

# %%
pred = nm.dft['prediction'].to_numpy()
pred = np.stack(pred)
pred.shape

# %%
pred[0]

# %%
y_test = nm.dft['yhat'].to_numpy()
y_pred = np.argmax(pred, axis=1)
y_pred

# %%
class_labels = ['real', 'var', 'lin', 'flaw', 'cr']

# %%

cm = confusion_matrix(y_test, y_pred)


# %%
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot()
plt.show()


# %%

label_binarizer = LabelBinarizer().fit(y_test)
y_onehot_test = label_binarizer.transform(y_test)
y_onehot_test.shape  # (n_samples, n_classes)

# %%
label_binarizer.classes_

# %%
n_classes=len(label_binarizer.classes_)
class_of_interest = "real" #0
class_id = np.flatnonzero(label_binarizer.classes_ == 0)[0]
class_id

# %%
RocCurveDisplay.from_predictions(
    y_onehot_test.ravel(),
    pred.ravel(),
    name="micro-average OvR",
    color="darkorange",
    #plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Micro-averaged One-vs-Rest\nReceiver Operating Characteristic")
plt.legend()
plt.show()

# %%
# store the fpr, tpr, and roc_auc for all averaging strategies
fpr, tpr, roc_auc = dict(), dict(), dict()
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], thresholds = roc_curve(y_onehot_test.ravel(), pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

# %%
optimal_threshold_idx = np.argmax(tpr["micro"] - fpr["micro"])
optimal_threshold = thresholds[optimal_threshold_idx]
print(optimal_threshold)


# %%
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr_grid = np.linspace(0.0, 1.0, 1000)

# Interpolate all ROC curves at these points
mean_tpr = np.zeros_like(fpr_grid)

for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

# Average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = fpr_grid
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

# %%
RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange",
    #plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\nReal vs Not")
plt.legend()
plt.show()

# %%
from itertools import cycle

fig, ax = plt.subplots(figsize=(6, 6))

plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for class_id, color in zip(range(n_classes), colors):
    RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        pred[:, class_id],
        name=f"ROC curve for {class_labels[class_id]}",
        color=color,
        ax=ax,
        #plot_chance_level=(class_id == 2),
    )

plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
plt.legend()
plt.show()

# %%
