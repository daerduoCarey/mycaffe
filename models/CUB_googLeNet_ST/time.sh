#!/usr/bin/env sh

build/tools/caffe time -model models/CUB_googLeNet_ST/train_val.prototxt -weights models/CUB_googLeNet_ST/init_CUB_googLeNet_ST_INC1_INC2.caffemodel -gpu 0 -iterations 5
