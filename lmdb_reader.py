import numpy as np
import matplotlib.pyplot as plt
import lmdb
import caffe
import sys

env = lmdb.open(sys.argv[1], readonly=True)

with env.begin() as txn:
	cursor = txn.cursor()
	for key, value in cursor:
		datum = caffe.proto.caffe_pb2.Datum()
		datum.ParseFromString(value)

		flat_x = np.fromstring(datum.data, dtype=np.uint8)
		x = flat_x.reshape(datum.channels, datum.height, datum.width)
		y = datum.label
	
		plt.title(y)
		print x.shape
		plt.imshow(x.transpose(1, 2, 0))
		plt.show()
