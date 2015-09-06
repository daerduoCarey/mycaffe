from caffe.proto import caffe_pb2
from caffe import Net
import sys

from google.protobuf import text_format

network = caffe_pb2.NetParameter()
network_bn = caffe_pb2.NetParameter()

USAGE = "USAGE: add_bn.python [input_filename] [output_filename]"
if len(sys.argv) != 3:
	print USAGE

f = open(sys.argv[1], 'r')
text_format.Merge(str(f.read()), network)
f.close

network_bn.name = network.name + "_bn"

k = 0
l = []

for layer in network.layer:
	if layer.type == "ReLU" or layer.type == "Sigmoid":
		k = k + 1
		new_bn_layer = network_bn.layer.add()
		new_bn_layer.type = 'BN'
		new_bn_layer.name = 'BN_' + '{:08}'.format(k)
		new_bn_layer.bottom.append(layer.bottom[0] + '')
		new_bn_layer.top.append(layer.bottom[0] + '/bn')
		l.append(layer.bottom[0] + '')
		print "Adding layer BN_" + '{:08}'.format(k) + " of type BN"

	for i in range(0, len(layer.bottom)):
		if layer.bottom[i] in l:
			layer.bottom[i] = layer.bottom[i] + '/bn'
	for i in range(0, len(layer.top)):
		if layer.top[i] in l:
			layer.top[i] = layer.top[i] + '/bn'
	print "Copying layer " + layer.name + " of type  " + layer.type
	network_bn.layer.extend([layer])

with open(sys.argv[2], 'w') as fout:
	fout.write(text_format.MessageToString(network_bn))

