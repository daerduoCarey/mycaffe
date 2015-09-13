#!/usr/bin/env sh

./build/tools/caffe train -solver models/inception_6_few_bn/solver.prototxt -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel -gpu 3
