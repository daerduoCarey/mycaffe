#!/bin/bash

CAFFE_ROOT=/data/kaichunm/mycaffe

$CAFFE_ROOT/build/tools/caffe time -model train_val.prototxt -weights snapshots/inception_bn_solver_stepsize_6400_iter_70.caffemodel -gpu 2 -iterations 5
