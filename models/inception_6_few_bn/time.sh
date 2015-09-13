#!/usr/bin/env sh

./build/tools/caffe time -model models/inception_6_few_bn/train_val.prototxt -weights models/inception_6_few_bn/caffemodels/inception_6_few_bn_iter_60000.caffemodel -gpu 3 -iterations 3
