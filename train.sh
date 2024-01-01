#!/bin/bash
# train models 
trainAll=true

# required parameters:
#  model
#  round 1
#  revision
#  transform
#  arch
#  batch_size
#  train | trainera
#  
# default parameter values:
#  trial 1
#  epoch 0
#  patience 0
#  threshold 1e-5
#  lr 1e-3
#  epsilon 1.0
#  epochs 20
#  datadir ./data
#  trainpat train*.tfr
#  validpat valid*.tfr
#  monitor val_loss

#python train.py --round 1 --epochs 50 --batch_size 256 --lr 1e-3 --epsilon 50.0 --monitor val_target_output_loss --model cigar_m82 --arch AE --transform 'NaN,-1,-1|YX,122,13' --train
#python train.py --epochs 150 --batch_size 512 --lr 1e-5 --epsilon 50.0 --monitor val_target_output_loss --model sunfl_m63 --arch AE --transform 'NaN,-1,-1|Pad,1586,1600,-1|YX,40,40' --train
#python train.py --epochs 50 --batch_size 512 --lr 1e-5 --epsilon 50.0 --monitor val_target_output_loss --model andro_m31 --arch AE --revision 434 --epoch 94 --transform 'NaN,-1,-1|Slice,0,1536|XY,48,32' --train
#andro_m31v1r434t3-e94
#python predict.py --model pinwh_m101 --version 1 --revision 434 --trial 1 --epoch 0 --batch_size 8192 --arch LGB --transform 'NaN,-1,-1' --etransform - -
#python predict.py --model trian_m33 --version 1 --revision 439 --trial 1 --epoch 0 --batch_size 8192 --arch XGB --transform 'NaN,-1,-1'
#python predict.py --model bodes_m81 --version 1 --revision 439 --trial 3 --epoch 2 --batch_size 1024 --arch NR --transform 'NaN,-1,-1'
#python predict.py --model whirl_m51 --version 1 --revision 441 --trial 2 --epoch 43 --batch_size 8192 --arch AE --transform 'NaN,-1,-1|Slice,0,1568|YXZ,14,14,8'
#python predict.py --model sombr_m104 --version 1 --revision 444 --trial 1 --epoch 11 --batch_size 8192 --arch NC --transform 'NaN,-1,-1|Pad,1586,1600,-1|YX,64,25|Sparse,4,0'
#python predict.py --model breccia --version 1 --revision 444 --trial 1 --epoch 0 --batch_size 8192 --arch LGB --transform 'NaN,-1,-1'
# by era
#python train.py --epochs 20 --batch_size 256 --lr 1e-3 --epsilon 0.7 --monitor val_loss --model andro_m31 --arch AE --version 1 --revision 434 --trial 3 --epoch '94 --transform 'NaN,-1,-1|Slice,0,1536|XY,48,32' --trainera
#python train.py --epochs 20 --batch_size 256 --lr 1e-3 --epsilon 0.7 --monitor val_loss --model whirl_m51 --arch AE --version 1 --revision 441 --trial 2 --epoch 43 --transform 'NaN,-1,-1|Slice,0,1568|YXZ,14,14,8' --trainera
#python train.py --epochs 20 --batch_size 256 --lr 1e-3 --monitor val_accuracy --model sombr_m104 --arch NC --version 1 --revision 444 --trial 1 --epoch 11 --transform 'NaN,-1,-1|Pad,1586,1600,-1|YX,40,40' --trainera

# Models:
#andro_m31v6r0t2-e0

#endro_m31	6	0	2	AE	0	50	256	1e-3	50.0	val_loss	all	0	NaN,-1,-1|Slice,0,1536|XY,48,32
# Autotrain Config:
cat <<EOF >train.conf
model		ver	rev	trial	arch	epoch	epochs	batch	lr	epsilon	monitor		mode	begin	transform
cnnmc		4	1	1	MC	0	20	32	5e-5	0.0	val_loss	all	0	YX,100,100|OneHot,5,0
EOF

#MODELS='Geode Pumice Schist Granite Diorite Gneiss Marble Gabbro Andesite Rhyolite Mariposite Slate Breccia Concretion'
#MODELS='diorite gabbro granite geode rhyolite pumice gneiss schist marble mariposite slate concretion andesite breccia'
#MODELS='sunfl_m63 andro_m31 pinwh_m101 trian_m33 bodes_m81 whirl_m51 sombr_m104'
#MODELS='bodes_m81 blackeye_m64'
MODELS='cnnmc'

for m in $MODELS
do
  grep "^$m" train.conf | while read model version rev trial arch epoch epochs batch lr epsilon monitor mode begin transform
  do
    case $mode in
      all)
        #python train.py --model $model --version $version --trial $trial --arch $arch --transform $transform --round 1 --epochs $epochs --batch_size $batch --lr $lr --epsilon $epsilon --monitor $monitor --tsh rain
        python train.py --begin $begin --model $model --version $version --revision $rev --trial $trial --epoch $epoch --arch $arch --transform $transform --round 1 --epochs $epochs --batch_size $batch --lr $lr --epsilon $epsilon --monitor $monitor --train
        ;;
      *)
        echo mode error in train.conf
        ;;
    esac
  done
done
