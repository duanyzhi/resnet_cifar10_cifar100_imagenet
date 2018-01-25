from __future__ import absolute_import  # 加入绝对引入这个新特性,便于导入系统中函数(和自己写的文件名一样的函数)
from __future__ import division  # 精确除法　导入精确除法后，若要执行截断除法，可以使用"//"操作符：3/4=0.75  3//4=0
from __future__ import print_function  # 使用新版本的print

from DeepLearning.deep_tensorflow import *
from lib.config.config import FLAGS
from lib.layer_utils.cnn_utils import create_variables, batch_norm

def Basic_Residual_block(in_img, filters, down_sample, projection=False, model='train'):   # 0.913
    input_depth = int(in_img.get_shape()[3])
    if down_sample:
        stride = 2
    else:
        stride = 1

    with tf.variable_scope('conv1'):
        weight_1 = create_variables(name='weight_1', shape=[3, 3, input_depth, filters])
        biases_1 = create_variables(name='biases_1', shape=[filters], initializer=tf.zeros_initializer())
        conv1_ReLu = tf.nn.relu(tf.nn.conv2d(in_img, weight_1, strides=[1, stride, stride, 1], padding='SAME') + biases_1)
        # conv1_ReLu = Conv_BN_Relu(in_img, weight_1, filters, strides=stride)

    with tf.variable_scope('conv2'):
        weight_2 = create_variables(name='weight_2', shape=[3, 3, filters, filters])
        biases_2 = create_variables(name='biases_2', shape=[filters], initializer=tf.zeros_initializer())
        conv2_ReLu = tf.nn.conv2d(conv1_ReLu, weight_2, strides=[1, 1, 1, 1], padding='SAME') + biases_2
        conv2_ReLu = batch_norm('bn', conv2_ReLu)
        # conv2_ReLu = Conv_BN_Relu(conv1_ReLu, weight1_2, biases_2, filters, strides=1, active=None)  # 这里不加relu!

    if input_depth != filters:
        if projection:   # Not very good
            # Option B: Projection shortcut
            weight_3 = create_variables(name='weight_3', shape=[1, 1, input_depth, filters])
            biases_3 = create_variables(name='biases_3', shape=[filters], initializer=tf.zeros_initializer())
            input_layer =  tf.nn.conv2d(in_img, weight_3, strides=[1, 1, 1, 1], padding='SAME') + biases_3
        else:
            # Option A: Zero-padding
            if down_sample:
                in_img = ave_pool(in_img)
            input_layer = tf.pad(in_img, [[0, 0], [0, 0], [0, 0], [int((filters - input_depth)/2), filters - input_depth - int((filters - input_depth)/2)]])  # 维度是4维[batch_size, :, :, dim] 我么要pad dim的维度
    else:
        input_layer = in_img

    output = conv2_ReLu + input_layer
    # output = tf.nn.relu(output)
    return output


"""
  resnet-cifar10
  in: [32, 32, 3]
  conv: [32， 32， 16]
  conv1 :[32, 32, 16*k]
  conv2: [16, 16, 32*k]
  conv3: [8, 8, 64*k]
  ave-pool : [8*8] pooling----[1*1]
  fc: [64*k, 10]
"""


class ResNet_cifar:  # Inference
    def __init__(self):
        self.img = None
        self.reuse = False
        self.k = FLAGS.depth_filter    # Original architecture
        self.filter = [16, 16*self.k, 32*self.k, 64*self.k]
        (self.stack_layers, rem) = divmod(FLAGS.layers_depth - 2, 6)
        assert rem == 0, 'depth must be 6n + 2, 余数为０'
        self.block = Basic_Residual_block
        # print(self.stack_layers)

    def __call__(self, img, scope):
        self.img = img
        with tf.variable_scope(scope, reuse=self.reuse) as scope_name:
            if self.reuse:
                scope_name.reuse_variables()
            # conv1
            with tf.variable_scope('conv_pre'):   # 32
                weight_1 = create_variables(name='weight', shape=[3, 3, 3, self.filter[0]])
                biases_1 = create_variables(name='biases', shape=[self.filter[0]], initializer=tf.zeros_initializer())
                conv1_conv = tf.nn.conv2d(self.img, weight_1, strides=[1, 1, 1, 1], padding='SAME') + biases_1
                conv1_BN = batch_norm('bn', conv1_conv)
                conv1 = tf.nn.relu(conv1_BN)
            # conv2
            in_img = conv1
            for ii in range(1, 4):
                with tf.variable_scope('conv' + str(ii)):   # 64
                    for kk in range(self.stack_layers):
                        down_sample = True if kk == 0 and ii != 1 else False
                        with tf.variable_scope('con_' + str(kk)):
                            in_img = self.block(in_img, filters=self.filter[ii], down_sample=down_sample)
            in_img = tf.nn.relu(in_img)
            with tf.variable_scope('fc'):
                fc_layer = batch_norm('bn', in_img)
                global_pool = tf.reduce_mean(fc_layer, [1, 2])

                weight_fc = create_variables('fc_weight', [self.filter[3], FLAGS.classes_number])
                biases_fc = create_variables('fc_biases', [FLAGS.classes_number], initializer=tf.zeros_initializer())
                fc = tf.add(tf.matmul(global_pool, weight_fc), biases_fc)   # fc的biases必需

                mean, variance = tf.nn.moments(fc, [0, 1])
                fc = tf.nn.batch_normalization(fc, mean, variance, 0, 1, FLAGS.epsilon)
        self.reuse = True
        return fc

# max: 0.930
