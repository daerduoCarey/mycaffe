"""Builds a version of the GoogLeNet (Inception) network for ILSVRC12."""

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=no-self-use
# pylint: disable=too-few-public-methods

from caffe.proto import caffe_pb2
from caffe import Net

from google.protobuf import text_format

# pylint: disable=invalid-name
# pylint: disable=no-member
LayerType = caffe_pb2.LayerParameter.LayerType
PoolMethod = caffe_pb2.PoolingParameter.PoolMethod
DBType = caffe_pb2.DataParameter.DB
# pylint: enable=invalid-name
# pylint: enable=no-member

class InceptionBuilder(object):
    """Builder class for Inception network design.
       Alter the design by subclassing."""

    def __init__(self, training_source, training_batch_size, testing_source,
                             testing_batch_size, task_name, **kwargs):
        self.training_source = training_source
        self.training_batch_size = training_batch_size
        self.testing_source = testing_source
        self.testing_batch_size = testing_batch_size
        self.task_name = task_name

        self.other_args = kwargs

    def _make_train_layer(self, network):
        """Make training data layer."""

        layer = network.layers.add()
        layer.name = 'train'

        layer.top.append('data')
        for label, _ in self.other_args['labels']:
            layer.top.append(label)

        layer.type = LayerType.Value('DATA')
        params = layer.data_param
        params.source = self.training_source
        params.batch_size = self.training_batch_size
        params.backend = DBType.Value('LMDB')

        transform = layer.transform_param
        transform.crop_size = 224
        transform.mean_file = self.other_args['mean_file']
        transform.mirror = True
        transform.scale = 1.0/128.0

        self._set_mode(layer, 'TRAIN')

        return layer

    def _make_test_layer(self, network):
        """Make testing data layer."""

        layer = network.layers.add()
        layer.name = 'test'

        layer.top.append('data')
        for label, _ in self.other_args['labels']:
            layer.top.append(label)

        layer.type = LayerType.Value('DATA')
        params = layer.data_param
        params.source = self.testing_source
        params.batch_size = self.testing_batch_size
        params.backend = DBType.Value('LMDB')

        transform = layer.transform_param
        transform.crop_size = 224
        transform.mean_file = self.other_args['mean_file']
        transform.mirror = False
        transform.scale = 1.0/128.0

        self._set_mode(layer, 'TEST')

        return layer

    def _set_mode(self, layer, mode):
        """Set layer for TRAIN or TEST mode."""

        include = layer.include.add()
        include.phase = caffe_pb2.Phase.Value(mode)

    def _make_conv_layer(self, network, kernel_size, num_output, stride=1,
                         pad=0, bias_value=0):
        """Make convolution layer."""

        layer = network.layers.add()
        layer.name = 'conv-%dx%d-%d' % (kernel_size, kernel_size, stride)

        layer.type = LayerType.Value('CONVOLUTION')
        params = layer.convolution_param
        params.num_output = num_output
        params.kernel_size = kernel_size
        params.stride = stride
        params.pad = pad
        weight_filler = params.weight_filler
        weight_filler.type = 'xavier'
        bias_filler = params.bias_filler
        bias_filler.type = 'constant'
        bias_filler.value = bias_value

        layer.blobs_lr.append(1)
        layer.blobs_lr.append(2)

        layer.weight_decay.append(1)
        layer.weight_decay.append(0)

        return layer

    def _make_maxpool_layer(self, network, kernel_size, stride=1):
        """Make max pooling layer."""

        layer = network.layers.add()
        layer.name = 'maxpool-%dx%d-%d' % (kernel_size, kernel_size, stride)

        layer.type = LayerType.Value('POOLING')
        params = layer.pooling_param
        params.pool = PoolMethod.Value('MAX')
        params.kernel_size = kernel_size
        params.stride = stride

        return layer

    def _make_avgpool_layer(self, network, kernel_size, stride=1):
        """Make average pooling layer."""

        layer = network.layers.add()
        layer.name = 'avgpool-%dx%d-%d' % (kernel_size, kernel_size, stride)

        layer.type = LayerType.Value('POOLING')
        params = layer.pooling_param
        params.pool = PoolMethod.Value('AVE')
        params.kernel_size = kernel_size
        params.stride = stride

        return layer

    def _make_lrn_layer(self, network):
        """Make local response normalization layer."""

        layer = network.layers.add()
        layer.name = 'lrn'

        layer.type = LayerType.Value('LRN')
        params = layer.lrn_param
        params.local_size = 5
        params.alpha = 0.0001
        params.beta = 0.75

        return layer

    def _make_concat_layer(self, network):
        """Make depth concatenation layer."""

        layer = network.layers.add()
        layer.name = 'concat'

        layer.type = LayerType.Value('CONCAT')

        return layer

    def _make_dropout_layer(self, network, dropout_ratio=0.5):
        """Make dropout layer."""

        layer = network.layers.add()
        layer.name = 'dropout'

        layer.type = LayerType.Value('DROPOUT')
        params = layer.dropout_param
        params.dropout_ratio = dropout_ratio

        return layer

    def _make_inner_product_layer(self, network, num_output, weight_lr=1,
                                  bias_lr=2, weight_std=1, bias_value=0):
        """Make inner product layer."""

        layer = network.layers.add()
        layer.name = 'inner_product'

        layer.type = LayerType.Value('INNER_PRODUCT')
        params = layer.inner_product_param
        params.num_output = num_output
        weight_filler = params.weight_filler
        weight_filler.type = 'gaussian'
        weight_filler.std = weight_std
        bias_filler = params.bias_filler
        bias_filler.type = 'constant'
        bias_filler.value = bias_value

        layer.blobs_lr.append(weight_lr)
        layer.blobs_lr.append(bias_lr)

        layer.weight_decay.append(1)
        layer.weight_decay.append(0)

        return layer

    def _make_softmax_loss_layer(self, network, loss_id, loss_weight=1):
        """Make softmax loss layer."""

        layer = network.layers.add()
        layer.name = 'softmax_loss'

        layer.type = LayerType.Value('SOFTMAX_LOSS')
        layer.loss_weight.append(loss_weight)

        self._set_mode(layer, 'TRAIN')

        layer.top.append('loss_%s' % loss_id)

        return layer

    def _make_accuracy_layer(self, network, accuracy_id):
        """Make accuracy layer for classification tasks."""

        layer = network.layers.add()
        layer.name = 'accuracy'

        layer.type = LayerType.Value('ACCURACY')

        self._set_mode(layer, 'TEST')

        layer.top.append('accuracy_%s' % accuracy_id)

        return layer

    def _make_split_layer(self, network):
        """Make split layer."""

        layer = network.layers.add()
        layer.name = 'split'

        layer.type = LayerType.Value('SPLIT')

        return layer

    def _make_relu_layer(self, network):
        """Make ReLU layer."""

        layer = network.layers.add()
        layer.name = 'relu'

        layer.type = LayerType.Value('RELU')

        return layer

    def _make_norm_layer(self, network):
        """Make batch normalization layer."""

        layer = network.layers.add()
        layer.name = 'bn'

        layer.type = LayerType.Value('DROMPNAR')

        layer.blobs_lr.append(1)
        layer.blobs_lr.append(1)
        layer.weight_decay.append(0)
        layer.weight_decay.append(0)

        return layer

    def _make_inception(self, network, x1x1, x3x3r, x3x3, x5x5r, x5x5, proj,
                        name_generator):
        """Make Inception submodule."""

        layers = []

        split = self._make_split_layer(network)
        layers.append(split)

        context1_norm = self._make_norm_layer(network)
        layers.append(context1_norm)

        context1 = self._make_conv_layer(network, kernel_size=1,
            num_output=x1x1)
        layers.append(context1)

        relu1 = self._make_relu_layer(network)
        layers.append(relu1)

        context2a_norm = self._make_norm_layer(network)
        layers.append(context2a_norm)

        context2a = self._make_conv_layer(network, kernel_size=1,
            num_output=x3x3r)
        layers.append(context2a)

        relu2a = self._make_relu_layer(network)
        layers.append(relu2a)

        context2b_norm = self._make_norm_layer(network)
        layers.append(context2b_norm)

        context2b = self._make_conv_layer(network, kernel_size=3,
            num_output=x3x3, pad=1)
        layers.append(context2b)

        relu2b = self._make_relu_layer(network)
        layers.append(relu2b)

        context3a_norm = self._make_norm_layer(network)
        layers.append(context3a_norm)

        context3a = self._make_conv_layer(network, kernel_size=1,
            num_output=x5x5r)
        layers.append(context3a)

        relu3a = self._make_relu_layer(network)
        layers.append(relu3a)

        context3b_norm = self._make_norm_layer(network)
        layers.append(context3b_norm)

        context3b = self._make_conv_layer(network, kernel_size=5,
            num_output=x5x5, pad=2)
        layers.append(context3b)

        relu3b = self._make_relu_layer(network)
        layers.append(relu3b)

        context4a = self._make_maxpool_layer(network, kernel_size=3)
        layers.append(context4a)

        relu4a = self._make_relu_layer(network)
        layers.append(relu4a)

        context4b_norm = self._make_norm_layer(network)
        layers.append(context4b_norm)

        context4b = self._make_conv_layer(network, kernel_size=1,
            num_output=proj, pad=1)
        layers.append(context4b)

        relu4b = self._make_relu_layer(network)
        layers.append(relu4b)

        concat = self._make_concat_layer(network)
        layers.append(concat)

        connections = [
            (split.name, (split.top, context1_norm.bottom)),
            (split.name, (split.top, context2a_norm.bottom)),
            (split.name, (split.top, context3a_norm.bottom)),
            (split.name, (split.top, context4a.bottom)),
            (context1_norm.name, (context1_norm.top, context1.bottom)),
            (context2a_norm.name, (context2a_norm.top, context2a.bottom)),
            (context3a_norm.name, (context3a_norm.top, context3a.bottom)),
            (context2a.name,
                (context2a.top, relu2a.bottom, relu2a.top,
                    context2b_norm.bottom)),
            (context3a.name,
                (context3a.top, relu3a.bottom, relu3a.top,
                    context3b_norm.bottom)),
            (context4a.name,
                (context4a.top, relu4a.bottom, relu4a.top,
                    context4b_norm.bottom)),
            (context1.name,
                (context1.top, relu1.bottom, relu1.top, concat.bottom)),
            (context2b_norm.name, (context2b_norm.top, context2b.bottom)),
            (context3b_norm.name, (context3b_norm.top, context3b.bottom)),
            (context4b_norm.name, (context4b_norm.top, context4b.bottom)),
            (context2b.name,
                (context2b.top, relu2b.bottom, relu2b.top, concat.bottom)),
            (context3b.name,
                (context3b.top, relu3b.bottom, relu3b.top, concat.bottom)),
            (context4b.name,
                (context4b.top, relu4b.bottom, relu4b.top, concat.bottom)),
        ]

        for connection in connections:
            self._tie(connection, name_generator)

        return layers

    def _make_tendril(self, network, tendril_id, name_generator):
        """Make internal classification network."""

        layers = []

        avgpool = self._make_avgpool_layer(network, kernel_size=5, stride=3)
        self._set_mode(avgpool, 'TRAIN')
        layers.append(avgpool)

        conv_norm = self._make_norm_layer(network)
        self._set_mode(conv_norm, 'TRAIN')
        layers.append(conv_norm)

        conv = self._make_conv_layer(network, kernel_size=1, stride=1,
            num_output=128)
        self._set_mode(conv, 'TRAIN')
        layers.append(conv)

        relu1 = self._make_relu_layer(network)
        self._set_mode(relu1, 'TRAIN')
        layers.append(relu1)

        inner_product1_norm = self._make_norm_layer(network)
        self._set_mode(inner_product1_norm, 'TRAIN')
        layers.append(inner_product1_norm)

        inner_product1 = self._make_inner_product_layer(network,
            num_output=1024, weight_std=0.03827327723)
        self._set_mode(inner_product1, 'TRAIN')
        layers.append(inner_product1)

        relu2 = self._make_relu_layer(network)
        self._set_mode(relu2, 'TRAIN')
        layers.append(relu2)

        dropout = self._make_dropout_layer(network, dropout_ratio=0.7)
        self._set_mode(dropout, 'TRAIN')
        layers.append(dropout)

        final_layers = self._make_tendril_classifiers(network, tendril_id,
            name_generator)
        layers += final_layers

        connections = [
            (avgpool.name, (avgpool.top, conv_norm.bottom)),
            (conv_norm.name, (conv_norm.top, conv.bottom)),
            (conv.name,
                (conv.top, relu1.bottom, relu1.top,
                    inner_product1_norm.bottom)),
            (inner_product1_norm.name,
                (inner_product1_norm.top, inner_product1.bottom)),
            (inner_product1.name,
                (inner_product1.top, relu2.bottom, relu2.top, dropout.bottom)),
            (dropout.name, (dropout.top, final_layers[0].bottom)),
        ]

        for connection in connections:
            self._tie(connection, name_generator)

        return layers

    def _make_tendril_classifiers(self, network, tendril_id, name_generator):
        """Builds loss layers for tendril."""

        layers = []

        split = self._make_split_layer(network)
        self._set_mode(split, 'TRAIN')
        layers.append(split)

        connections = []

        softmaxes = []
        for label, num_output in self.other_args['labels']:
            inner_product2_norm = self._make_norm_layer(network)
            self._set_mode(inner_product2_norm, 'TRAIN')
            layers.append(inner_product2_norm)

            inner_product2 = self._make_inner_product_layer(network,
                num_output=num_output, weight_std=0.05412658773/32)
            self._set_mode(inner_product2, 'TRAIN')
            layers.append(inner_product2)

            softmax_loss = self._make_softmax_loss_layer(network,
                loss_id='%s_tendril%d' % (label, tendril_id))
            layers.append(softmax_loss)

            connections.append((split.name,
                (split.top, inner_product2_norm.bottom)))
            connections.append((inner_product2_norm.name,
                (inner_product2_norm.top, inner_product2.bottom)))
            connections.append((inner_product2.name,
                (inner_product2.top, softmax_loss.bottom)))

            softmaxes.append((label, softmax_loss))

        for connection in connections:
            self._tie(connection, name_generator)

        for label, loss in softmaxes:
            loss.bottom.append(label)

        return layers

    def _make_final_classifiers(self, network, name_generator):
        """Builds loss and accuracy evaluation layers at end of network."""

        layers = []

        split = self._make_split_layer(network)
        layers.append(split)

        connections = []

        softmaxes = []
        for label, num_output in self.other_args['labels']:
            inner_product_norm = self._make_norm_layer(network)
            layers.append(inner_product_norm)

            inner_product = self._make_inner_product_layer(network,
                num_output=num_output, weight_std=0.05412658773/32)
            layers.append(inner_product)

            softmax_loss = self._make_softmax_loss_layer(network, loss_id=label)
            layers.append(softmax_loss)

            accuracy = self._make_accuracy_layer(network, accuracy_id=label)
            layers.append(accuracy)

            connections.append((split.name,
                (split.top, inner_product_norm.bottom)))
            connections.append((inner_product_norm.name,
                (inner_product_norm.top, inner_product.bottom)))
            connections.append((inner_product.name,
                (inner_product.top, softmax_loss.bottom, accuracy.bottom)))

            softmaxes.append((label, softmax_loss))
            softmaxes.append((label, accuracy))

        for connection in connections:
            self._tie(connection, name_generator)

        for label, loss in softmaxes:
            loss.bottom.append(label)

        return layers

    def _tie(self, layers, name_generator):
        """Generate a named connection between layer endpoints."""

        name = '%s-%d' % (layers[0], name_generator.next())
        for layer in layers[1]:
            layer.append(name)

    def _connection_name_generator(self):
        """Generate a unique id."""

        index = 0
        while True:
            yield index
            index += 1

    def _build_network(self):
        """Build the Inception network."""

        network = caffe_pb2.NetParameter()
        network.name = 'inception-%s' % self.task_name

        layers = []

        name_generator = self._connection_name_generator()

        train_layer = self._make_train_layer(network)
        layers.append(train_layer)
        test_layer = self._make_test_layer(network)
        layers.append(test_layer)

        conv1_norm = self._make_norm_layer(network)
        layers.append(conv1_norm)

        conv1 = self._make_conv_layer(network, kernel_size=7, stride=2,
            num_output=64, pad=3)
        layers.append(conv1)
        relu1 = self._make_relu_layer(network)
        layers.append(relu1)

        maxpool1 = self._make_maxpool_layer(network, kernel_size=3, stride=2)
        layers.append(maxpool1)
        lrn1 = self._make_lrn_layer(network)
        layers.append(lrn1)

        conv2_norm = self._make_norm_layer(network)
        layers.append(conv2_norm)

        conv2 = self._make_conv_layer(network, kernel_size=1, num_output=64)
        layers.append(conv2)
        relu2 = self._make_relu_layer(network)
        layers.append(relu2)

        conv3_norm = self._make_norm_layer(network)
        layers.append(conv3_norm)

        conv3 = self._make_conv_layer(network, kernel_size=3, num_output=64,
            pad=1)
        layers.append(conv3)
        relu3 = self._make_relu_layer(network)
        layers.append(relu3)

        lrn2 = self._make_lrn_layer(network)
        layers.append(lrn2)
        maxpool2 = self._make_maxpool_layer(network, kernel_size=3, stride=2)
        layers.append(maxpool2)

        inception1 = self._make_inception(network, 64, 96, 128, 16, 32, 32,
            name_generator)
        layers += inception1

        inception2 = self._make_inception(network, 128, 128, 192, 32, 96, 64,
            name_generator)
        layers += inception2

        maxpool3 = self._make_maxpool_layer(network, kernel_size=3, stride=2)
        layers.append(maxpool3)

        inception3 = self._make_inception(network, 192, 96, 208, 16, 48, 64,
            name_generator)
        layers += inception3

        inception4 = self._make_inception(network, 160, 112, 224, 24, 64, 64,
            name_generator)
        layers += inception4

        inception5 = self._make_inception(network, 128, 128, 256, 24, 64, 64,
            name_generator)
        layers += inception5

        inception6 = self._make_inception(network, 112, 144, 288, 32, 64, 64,
            name_generator)
        layers += inception6

        inception7 = self._make_inception(network, 256, 160, 320, 32, 128, 128,
            name_generator)
        layers += inception7

        maxpool4 = self._make_maxpool_layer(network, kernel_size=3, stride=2)
        layers.append(maxpool4)

        inception8 = self._make_inception(network, 256, 160, 320, 32, 128, 128,
            name_generator)
        layers += inception8

        inception9 = self._make_inception(network, 384, 192, 384, 48, 128, 128,
            name_generator)
        layers += inception9

        avgpool = self._make_avgpool_layer(network, kernel_size=7)
        layers.append(avgpool)

        dropout = self._make_dropout_layer(network, dropout_ratio=0.4)
        layers.append(dropout)

        tendril1 = self._make_tendril(network, 1, name_generator)
        layers += tendril1

        tendril2 = self._make_tendril(network, 2, name_generator)
        layers += tendril2

        final_layers = self._make_final_classifiers(network, name_generator)
        layers += final_layers

        connections = [
            (conv1_norm.name, (conv1_norm.top, conv1.bottom)),
            (conv1.name, (conv1.top, relu1.bottom, relu1.top, maxpool1.bottom)),
            (maxpool1.name, (maxpool1.top, lrn1.bottom)),
            (lrn1.name, (lrn1.top, conv2_norm.bottom)),
            (conv2_norm.name, (conv2_norm.top, conv2.bottom)),
            (conv2.name,
                (conv2.top, relu2.bottom, relu2.top, conv3_norm.bottom)),
            (conv3_norm.name, (conv3_norm.top, conv3.bottom)),
            (conv3.name, (conv3.top, relu3.bottom, relu3.top, lrn2.bottom)),
            (lrn2.name, (lrn2.top, maxpool2.bottom)),
            (maxpool2.name, (maxpool2.top, inception1[0].bottom)),
            (inception1[-1].name, (inception1[-1].top, inception2[0].bottom)),
            (inception2[-1].name, (inception2[-1].top, maxpool3.bottom)),
            (maxpool3.name, (maxpool3.top, inception3[0].bottom)),
            (inception3[-1].name,
                (inception3[-1].top, inception4[0].bottom, tendril1[0].bottom)),
            (inception4[-1].name, (inception4[-1].top, inception5[0].bottom)),
            (inception5[-1].name, (inception5[-1].top, inception6[0].bottom)),
            (inception6[-1].name,
                (inception6[-1].top, inception7[0].bottom, tendril2[0].bottom)),
            (inception7[-1].name, (inception7[-1].top, maxpool4.bottom)),
            (maxpool4.name, (maxpool4.top, inception8[0].bottom)),
            (inception8[-1].name, (inception8[-1].top, inception9[0].bottom)),
            (inception9[-1].name, (inception9[-1].top, avgpool.bottom)),
            (avgpool.name, (avgpool.top, dropout.bottom)),
            (dropout.name, (dropout.top, final_layers[0].bottom)),
        ]

        for connection in connections:
            self._tie(connection, name_generator)

        conv1_norm.bottom.append('data')

        for pos, layer in enumerate(layers):
            print layer
            layer.name += '-%d' % pos

        return network

    def build(self):
        """main method."""

        network = self._build_network()
        print network
        network_filename = 'inception.prototxt'
        with open(network_filename, 'w') as network_file:
            network_file.write(text_format.MessageToString(network))
        net = Net(network_filename)
        net.set_phase_test()
        net.set_mode_gpu()

        print net.forward()

if __name__ == '__main__':
    __inception_builder__ = InceptionBuilder(
        training_source='/ssd/ilsvrc2012/ilsvrc12_train_lmdb',
        #training_source='examples/imagenet/ilsvrc12_train_lmdb',
        training_batch_size=128,
        testing_source='/ssd/ilsvrc2012/ilsvrc12_val_lmdb',
        #testing_source='examples/imagenet/ilsvrc12_val_lmdb',
        testing_batch_size=50,
        task_name='ilsvrc12',
        mean_file='data/ilsvrc12/imagenet_mean.binaryproto',
        labels=[('class_id', 1000)],
    )

    __inception_builder__.build()
