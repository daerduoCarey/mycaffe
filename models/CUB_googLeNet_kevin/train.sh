#!/usr/bin/env sh

./build/tools/caffe train -solver models/CUB_googLeNet_kevin/solver.prototxt -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel -gpu 1
