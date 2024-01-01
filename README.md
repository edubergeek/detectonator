# detectonator
NEO detector for ATLAS FITS postage stamp images

## Usage
Run python etl.py to transform ATLAS postage stamp training set to Tensorflow TFRecord files.
Edit or copy cnnmc.py to configure a neural network model.
Run python cnnmc.py to construct and save the model (no weights).
Edit train.sh to select parameters to train the model.
Run train.sh to initiate a training run, saving best model weights.

## Directories
-  ../V1 - Subdirectory names are classes with a jpg file for each stamp image example of that class
  -    real - real NEO image
  -    var  - variable star ?
  -    lin  - CCD readout saturation line
  -    flaw - CCD known image flaw
  -    cr   - cosmic ray
-  ../Vr2 - Subdirectory names are classes with a jpg file for each stamp image example of that class
  -    real - real NEO image
  -    var  - variable star ?
  -    lin  - CCD readout saturation line
  -    flaw - CCD known image flaw
  -    cr   - cosmic ray
-  ../FITStamp - 100x100 postage stamp images
-  ../data - Output directory for etl.py of TFRecord shard files and Input dir for train.py
-  ./cnnmc - Each unique model name should save its models and weights to a directory of the same name

## Current Version

```
Untrained model: cnnmcv4r1t1-e0
Trained model:   cnnmcv4r1t1-e17
URL:             http://dtn-itc.ifa.hawaii.edu/atlas/detectonator/cnnmc
```

