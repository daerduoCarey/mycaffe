#!/usr/bin/env sh

./build/tools/caffe train -solver examples/mnist/fcn_solver.prototxt -gpu 2
