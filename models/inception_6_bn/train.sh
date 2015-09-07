#!/usr/bin/env sh

./build/tools/caffe train -solver models/inception_6_bn/solver.prototxt -gpu 0
