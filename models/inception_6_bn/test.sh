#!/usr/bin/env sh

./build/tools/caffe test -model models/inception_6_bn/train_val.prototxt -gpu 3 -iterations 5
