#!/usr/bin/env sh

./build/tools/caffe test -model models/CUB_googLeNet_kevin/train_val.prototxt -weights models/CUB_googLeNet_kevin/caffemodels/CUB_googLeNet_kevin_iter_2000.caffemodel -gpu 0 -iterations 200
