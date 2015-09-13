#!/usr/bin/env sh

./build/tools/caffe train -solver models/inception_6_bn_final/solver.prototxt -weights models/inception_6_bn_final/caffemodels/inception_6_bn_final_iter_32000.caffemodel -gpu 3
