#!/usr/bin/env sh

./build/tools/caffe train -solver models/CUB_googLeNet/solver.prototxt -gpu 0
