#!/usr/bin/env sh

./build/tools/caffe train -solver models/inception_6/solver.prototxt -gpu 1
