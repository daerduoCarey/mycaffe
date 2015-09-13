#!/usr/bin/env sh

./build/tools/caffe time -model models/inception_6_bn_final/train_val.prototxt -weights models/inception_6_bn_final/caffemodels/inception_6_bn_final_iter_60000.caffemodel -gpu 2 -iterations 5
