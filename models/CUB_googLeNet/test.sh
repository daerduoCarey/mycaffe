#!/usr/bin/env sh

./build/tools/caffe test -model models/CUB_googLeNet/train_test.prototxt -weights models/CUB_googLeNet/caffemodels/CUB_googLeNet_iter_74000.caffemodel -gpu 0 -iterations 91
