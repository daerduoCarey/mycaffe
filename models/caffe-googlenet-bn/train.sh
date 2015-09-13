#!/bin/bash

CAFFE_ROOT=/data/kaichunm/mycaffe

$CAFFE_ROOT/build/tools/caffe train -solver solver.prototxt -gpu 2
