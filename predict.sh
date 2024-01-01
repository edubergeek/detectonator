#!/bin/bash

MODELS='cnnmc'

for m in $MODELS
do
  #model      arch version round epoch transform	e1_model,arch,ver	e2_model,arch,ver
  grep "^$m" predict.conf | while read model path arch version revision trial epoch batch transform
  do
    echo python predict.py --path $path --model $model --version $version --revision $revision --trial $trial --epoch $epoch --batch_size $batch --arch $arch --transform \'$transform\'
    python predict.py --path $path --model $model --version $version --revision $revision --trial $trial --epoch $epoch --batch_size $batch --arch $arch --transform $transform
    #bash distrib.sh $model
  done
done
