#!/usr/bin/env sh

./build/tools/caffe time -model models/inception_6_bn_final/train_val.prototxt -weights models/inception_6_bn_final/caffemodels/inception_6_bn_final_iter_32000.caffemodel -gpu 3 -iterations 10
