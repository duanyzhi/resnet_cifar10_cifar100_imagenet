import tensorflow as tf
import lib.config.config as cfg
from DeepLearning.deep_tensorflow import *
from lib.layer_utils.cnn_utils import create_variables, batch_norm


def Residual_block(in_img, filters, projection=False):
    input_depth = int(in_img.get_shape()[3])

    with tf.variable_scope('conv1'):  # 1*1
        weight1 = create_variables(name='weight_1', shape=[1, 1, input_depth, filters[0]])
        biases1 = create_variables(name='biases_1', shape=[filters[0]], initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(in_img, weight1, strides=[1, 1, 1, 1], padding='SAME') + biases1
        bn = batch_norm('bn', conv)
        relu = tf.nn.relu(bn)

    with tf.variable_scope('conv2'):  # 3*3
        weight2 = create_variables(name='weight_2', shape=[3, 3, filters[0], filters[1]])
        biases2 = create_variables(name='biases_2', shape=[filters[1]], initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(relu, weight2, strides=[1, 1, 1, 1], padding='SAME') + biases2
        bn = batch_norm('bn', conv)
        relu = tf.nn.relu(bn)
        # conv2_ReLu = batch_norm('bn', conv2_ReLu)
        # conv2_ReLu = Conv_BN_Relu(conv1_ReLu, weight2, biases2, filters[1], strides=1)

    with tf.variable_scope('conv3'):  # 1*1
        weight3 = create_variables(name='weight_3', shape=[1, 1, filters[1], filters[2]])
        biases3 = create_variables(name='biases_3', shape=[filters[2]], initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(relu, weight3, strides=[1, 1, 1, 1], padding='SAME') + biases3
        out = batch_norm('bn', conv)
        # conv3_ReLu = Conv_BN_Relu(conv2_ReLu, weight3, biases3, filters[2], strides=1)

    if input_depth != filters[2]:
        if projection:
            # Option B: Projection shortcut
            weight_4 = create_variables(name='weight_4', shape=[1, 1, input_depth, filters[2]])
            biases_4 = create_variables(name='biases_4', shape=[filters[2]], initializer=tf.zeros_initializer())
            input_layer = Conv_BN_Relu(in_img, weight_4,  filters[2], biases=biases_4, strides=1)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(in_img, [[0, 0], [0, 0], [0, 0], [int((filters[2] - input_depth)/2), filters[2] - input_depth - int((filters[2] - input_depth)/2)]])  # 维度是4维[batch_size, :, :, dim] 我么要pad dim的维度
    else:
        input_layer = in_img

    output = out + input_layer
    output = tf.nn.relu(output)
    return output


"""
  in: [224, 224, 3]
       [112, 112, 64]
  conv1 :[56, 56, 64]
  conv2: [10, 10, 256]
  conv3: [5, 5, 512]
  conv4: [2, 2, 1024]
  conv5: [1, 1, 2048]
  fc: []
"""


class ResNet_ImageNet:
    def __init__(self):
        self.img = None
        self.reuse = False
        self.learning_rate = cfg.FLAGS.learning_rate

    def __call__(self, img, scope):
        self.img = img         # [224, 224, 3]
        with tf.variable_scope(scope, reuse=self.reuse) as scope_name:
            if self.reuse:
                scope_name.reuse_variables()
            # conv1
            with tf.variable_scope('conv1'):
                weight_1 = create_variables(name='weight', shape=[7, 7, 3, 64])
                biases_1 = create_variables(name='biases', shape=[64], initializer=tf.zeros_initializer())
                conv1_ReLu = tf.nn.conv2d(self.img, weight_1, strides=[1, 2, 2, 1], padding='SAME') + biases_1
                conv1_BN = batch_norm('bn', conv1_ReLu)
                conv1 = tf.nn.relu(conv1_BN)
                conv1 = max_pool(conv1, k_size=(3, 3), stride=(2, 2), pad='SAME')   # out [56, 56, 64]
            in_img = conv1
            # conv2
            with tf.variable_scope('conv2'):
                for kk in range(3):
                    with tf.variable_scope('Residual_' + str(kk)):
                        in_img = Residual_block(in_img, [64, 64, 256])    # [56, 56, 256]
            conv2 = max_pool(in_img, k_size=(2, 2), stride=(2, 2), pad='SAME')    # [28, 28, 256]
            in_img = conv2
            with tf.variable_scope('conv3'):
                for kk in range(4):
                    with tf.variable_scope('Residual_' + str(kk)):
                        in_img = Residual_block(in_img, [128, 128, 512])  # [28, 28, 512]
            conv3 = max_pool(in_img, k_size=(2, 2), stride=(2, 2), pad='SAME')    # [14, 14, 512]
            in_img = conv3
            with tf.variable_scope('conv4'):
                for kk in range(23):
                    with tf.variable_scope('Residual_' + str(kk)):
                        in_img = Residual_block(in_img, [256, 256, 1024])  # [14, 14, 1024]
            conv4 = max_pool(in_img, k_size=(2, 2), stride=(2, 2), pad='SAME')    # [7, 7, 1024]
            in_img = conv4
            with tf.variable_scope('conv5'):
                for kk in range(3):
                    with tf.variable_scope('Residual_' + str(kk)):
                        in_img = Residual_block(in_img, [512, 512, 2048])  # [7, 7, 2048]
            conv5 = ave_pool(in_img, k_size=(7, 7), stride=(1, 1))    # [1, 1, 2048]
            ave_pooling = tf.squeeze(conv5, [1, 2])
            with tf.variable_scope('fc'):
                weight_fc = create_variables('fc_weight', [2048, cfg.FLAGS.classes_number])
                biases_fc = create_variables('fc_biases', [cfg.FLAGS.classes_number], initializer=tf.zeros_initializer())
                fc = tf.add(tf.matmul(ave_pooling, weight_fc), biases_fc)  # fc的biases必需

                mean, variance = tf.nn.moments(fc, [0, 1])
                fc = tf.nn.batch_normalization(fc, mean, variance, 0, 1, cfg.FLAGS.epsilon)
        self.reuse = True
        return fc





