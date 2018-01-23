import tensorflow as tf
import numpy as np
import lib.config.config as cfg

"""
VGG_16:
    input:[224, 224, 3]  大小224*224的RGB图像         [84, 84, 3]
    convolution1:
          convolution1_1:
                weights_size:[3, 3, 3, 64]  卷积核大小3*3 输入RGB三维，输出64维
                out_put_size: [224, 224, 64]          [84, 84, 64]
          convolution1_2:
                weights_size:[3, 3, 64, 64]  卷积核大小3*3 64维，输出64维
                out_put_size: [224, 224, 64]          [84, 84, 64]
          pooling_1:
                 stride = 2   做一个2*2最大池化
                 out_put_size: [112, 112, 64]
    convolution2:
          convolution2_1:
                weights_size:[3, 3, 64, 128]  卷积核大小3*3 64，输出128维
                out_put_size: [112, 112, 128]         [42, 42, 128]
          convolution2_2:
                weights_size:[3, 3, 128, 128]  卷积核大小3*3 128维，输出128维
                out_put_size: [112, 112, 128]         [42, 42, 128]
          pooling_2:
                 stride = 2   做一个2*2最大池化
                 out_put_size: [56, 56, 128]          [21, 21, 128]
    convolution3:
          convolution3_1:
                weights_size:[3, 3, 128, 256]
                out_put_size: [56, 56, 256]
          convolution3_2:
                weights_size:[3, 3, 256, 256]
                out_put_size: [56, 56, 256]
          pooling_3:
                 stride = 2   做一个2*2最大池化
                 out_put_size: [28, 28, 256]
    convolution4:
          convolution4_1:
                weights_size:[3, 3, 256, 512]
                out_put_size: [28, 28, 512]
          convolution4_2:
                weights_size:[3, 3, 512, 512]
                out_put_size: [28, 28, 512]
          convolution4_3:
                weights_size:[3, 3, 512, 512]
                out_put_size: [28, 38, 512]
          pooling_4:
                 stride = 2   做一个2*2最大池化
                 out_put_size: [14, 14, 512]
    convolution5:
          convolution5_1:
                weights_size:[3, 3, 512, 512]
                out_put_size: [14, 14, 512]
          convolution5_2:
                weights_size:[3, 3, 512, 512]
                out_put_size: [14, 14, 512]
          convolution5_3:
                weights_size:[3, 3, 512, 512]
                out_put_size: [14, 14, 512]
          pooling_5:
                 stride = 2   做一个2*2最大池化
                 out_put_size: [7, 7, 512]              [3, 3, 10]
    reshape: 7*7*512=25088-------4096
    fully connected_1:
          in_put -- out_put = 25088 -- 4096
    fully connected_2:
          in_put -- out_put = 4096 -- 4096       4096 -- 1000
    fully connected_3:
          in_put -- out_put = 4096 -- 1000       1000 -- 10
    softmax(不是sigmoid):
          1000 -- 1000
"""


# 定义卷积，卷积后尺寸不变
def conv(input, weight, biases, offset, scale, strides=1):
    conv_conv = tf.nn.conv2d(input, weight, strides=[1, strides, strides, 1], padding='SAME') + biases
    mean, variance = tf.nn.moments(conv_conv, [0, 1, 2])
    conv_batch = tf.nn.batch_normalization(conv_conv, mean, variance, offset, scale, 1e-10)
    return tf.nn.relu(conv_batch)


# 池化，大小k*k
def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='VALID')


class vgg16:
    def __init__(self):
        self.learning_rate = 0.001  # 超参数
        self.reuse = False
        # self.parameters = []

    def vgg(self, imgs, scope):
        self.imgs = imgs
        # self.parameters = []

        with tf.variable_scope(scope, reuse=self.reuse) as scope_name:
            if self.reuse: scope_name.reuse_variables()
            # conv1_1
            with tf.variable_scope('conv1'):
                with tf.variable_scope('conv1_1'):
                    self.weight_1_1 = tf.get_variable('weight', [3, 3, 3, 64])
                    self.biases_1_1 = tf.get_variable('biases', [64])
                    self.offset_1_1 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                    self.scale_1_1 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                    conv1_relu_1_1 = conv(self.imgs, self.weight_1_1, self.biases_1_1, self.offset_1_1, self.scale_1_1, strides=1)
                    # conv1_1 = max_pool(conv1_relu_1_1, ksize=(2, 2), stride=(2, 2))
                with tf.variable_scope('conv1_2'):
                    self.weight_1_2 = tf.get_variable('weight', [3, 3, 64, 64])
                    self.biases_1_2 = tf.get_variable('biases', [64])
                    self.offset_1_2 = tf.get_variable('offset', [64], initializer=tf.constant_initializer(0.0))
                    self.scale_1_2 = tf.get_variable('scale', [64], initializer=tf.constant_initializer(1.0))
                    conv1_relu_1_2 = conv(conv1_relu_1_1, self.weight_1_2, self.biases_1_2, self.offset_1_2, self.scale_1_2,
                                        strides=1)
                    conv1_2 = max_pool(conv1_relu_1_2, ksize=(2, 2), stride=(2, 2))
            with tf.variable_scope('conv2'):
                with tf.variable_scope('conv2_1'):
                    self.weight_2_1 = tf.get_variable('weight', [3, 3, 64, 128])
                    self.biases_2_1 = tf.get_variable('biases', [128])
                    self.offset_2_1 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                    self.scale_2_1 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                    conv1_relu_2_1 = conv(conv1_2, self.weight_2_1, self.biases_2_1, self.offset_2_1,
                                          self.scale_2_1, strides=1)
                    # conv2_1 = max_pool(conv1_relu_2_1, ksize=(2, 2), stride=(2, 2))
                with tf.variable_scope('conv2_2'):
                    self.weight_2_2 = tf.get_variable('weight', [3, 3, 128, 128])
                    self.biases_2_2 = tf.get_variable('biases', [128])
                    self.offset_2_2 = tf.get_variable('offset', [128], initializer=tf.constant_initializer(0.0))
                    self.scale_2_2 = tf.get_variable('scale', [128], initializer=tf.constant_initializer(1.0))
                    conv1_relu_2_2 = conv(conv1_relu_2_1, self.weight_2_2, self.biases_2_2, self.offset_2_2,
                                          self.scale_2_2,
                                          strides=1)
                    conv2_2 = max_pool(conv1_relu_2_2, ksize=(2, 2), stride=(2, 2))
            with tf.variable_scope('conv3'):
                with tf.variable_scope('conv3_1'):
                    self.weight_3_1 = tf.get_variable('weight', [3, 3, 128, 256])
                    self.biases_3_1 = tf.get_variable('biases', [256])
                    self.offset_3_1 = tf.get_variable('offset', [256], initializer=tf.constant_initializer(0.0))
                    self.scale_3_1 = tf.get_variable('scale', [256], initializer=tf.constant_initializer(1.0))
                    conv1_relu_3_1 = conv(conv2_2, self.weight_3_1, self.biases_3_1, self.offset_3_1,
                                          self.scale_3_1, strides=1)
                    # conv2_1 = max_pool(conv1_relu_2_1, ksize=(2, 2), stride=(2, 2))
                with tf.variable_scope('conv3_2'):
                    self.weight_3_2 = tf.get_variable('weight', [3, 3, 256, 256])
                    self.biases_3_2 = tf.get_variable('biases', [256])
                    self.offset_3_2 = tf.get_variable('offset', [256], initializer=tf.constant_initializer(0.0))
                    self.scale_3_2 = tf.get_variable('scale', [256], initializer=tf.constant_initializer(1.0))
                    conv1_relu_3_2 = conv(conv1_relu_3_1, self.weight_3_2, self.biases_3_2, self.offset_3_2,
                                          self.scale_3_2,
                                          strides=1)
                    # conv3_2 = max_pool(conv1_relu_3_2, ksize=(2, 2), stride=(2, 2))
                with tf.variable_scope('conv3_3'):
                    self.weight_3_3 = tf.get_variable('weight', [3, 3, 256, 256])
                    self.biases_3_3 = tf.get_variable('biases', [256])
                    self.offset_3_3 = tf.get_variable('offset', [256], initializer=tf.constant_initializer(0.0))
                    self.scale_3_3 = tf.get_variable('scale', [256], initializer=tf.constant_initializer(1.0))
                    conv1_relu_3_3 = conv(conv1_relu_3_2, self.weight_3_3, self.biases_3_3, self.offset_3_3,
                                          self.scale_3_3,
                                          strides=1)
                    conv3_3 = max_pool(conv1_relu_3_3, ksize=(2, 2), stride=(2, 2))
            with tf.variable_scope('conv4'):
                with tf.variable_scope('conv4_1'):
                    self.weight_4_1 = tf.get_variable('weight', [3, 3, 256, 512])
                    self.biases_4_1 = tf.get_variable('biases', [512])
                    self.offset_4_1 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                    self.scale_4_1 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                    conv1_relu_4_1 = conv(conv3_3, self.weight_4_1, self.biases_4_1, self.offset_4_1,
                                          self.scale_4_1, strides=1)
                    # conv2_1 = max_pool(conv1_relu_2_1, ksize=(2, 2), stride=(2, 2))
                with tf.variable_scope('conv4_2'):
                    self.weight_4_2 = tf.get_variable('weight', [3, 3, 512, 512])
                    self.biases_4_2 = tf.get_variable('biases', [512])
                    self.offset_4_2 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                    self.scale_4_2 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                    conv1_relu_4_2 = conv(conv1_relu_4_1, self.weight_4_2, self.biases_4_2, self.offset_4_2,
                                          self.scale_4_2,
                                          strides=1)
                    # conv3_2 = max_pool(conv1_relu_3_2, ksize=(2, 2), stride=(2, 2))
                with tf.variable_scope('conv4_3'):
                    self.weight_4_3 = tf.get_variable('weight', [3, 3, 512, 512])
                    self.biases_4_3 = tf.get_variable('biases', [512])
                    self.offset_4_3 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                    self.scale_4_3 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                    conv1_relu_4_3 = conv(conv1_relu_4_2, self.weight_4_3, self.biases_4_3, self.offset_4_3,
                                          self.scale_4_3,
                                          strides=1)
                    conv4_3 = max_pool(conv1_relu_4_3, ksize=(2, 2), stride=(2, 2))
            with tf.variable_scope('conv5'):
                with tf.variable_scope('conv5_1'):
                    self.weight_5_1 = tf.get_variable('weight', [3, 3, 512, 512])
                    self.biases_5_1 = tf.get_variable('biases', [512])
                    self.offset_5_1 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                    self.scale_5_1 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                    conv1_relu_5_1 = conv(conv4_3, self.weight_5_1, self.biases_5_1, self.offset_5_1,
                                          self.scale_5_1, strides=1)
                    # conv2_1 = max_pool(conv1_relu_2_1, ksize=(2, 2), stride=(2, 2))
                with tf.variable_scope('conv5_2'):
                    self.weight_5_2 = tf.get_variable('weight', [3, 3, 512, 512])
                    self.biases_5_2 = tf.get_variable('biases', [512])
                    self.offset_5_2 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                    self.scale_5_2 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                    conv1_relu_5_2 = conv(conv1_relu_5_1, self.weight_5_2, self.biases_5_2, self.offset_5_2,
                                          self.scale_5_2,
                                          strides=1)
                    # conv3_2 = max_pool(conv1_relu_3_2, ksize=(2, 2), stride=(2, 2))
                with tf.variable_scope('conv5_3'):
                    self.weight_5_3 = tf.get_variable('weight', [3, 3, 512, 512])
                    self.biases_5_3 = tf.get_variable('biases', [512])
                    self.offset_5_3 = tf.get_variable('offset', [512], initializer=tf.constant_initializer(0.0))
                    self.scale_5_3 = tf.get_variable('scale', [512], initializer=tf.constant_initializer(1.0))
                    conv1_relu_5_3 = conv(conv1_relu_5_2, self.weight_5_3, self.biases_5_3, self.offset_5_3,
                                          self.scale_5_3,
                                          strides=1)
                    conv5_3 = max_pool(conv1_relu_5_3, ksize=(2, 2), stride=(2, 2))
            with tf.variable_scope('fc'):
                with tf.variable_scope('fc1'):
                    shape = int(np.prod(conv5_3.get_shape()[1:]))
                    self.weight_fc_1 = tf.get_variable('weight', [shape, 4096])
                    self.biases_fc_1 = tf.get_variable('biases', [4096])
                    pool5_flat = tf.reshape(conv5_3, [-1, shape])
                    fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, self.weight_fc_1), self.biases_fc_1)
                    # self.offset_fc_1 = tf.get_variable('offset', [4096], initializer=tf.constant_initializer(0.0))
                    # self.scale_fc1 = tf.get_variable('scale', [4096], initializer=tf.constant_initializer(1.0))
                    # mean, variance = tf.nn.moments(fc1l, [0, 1, 2])
                    # conv_batch = tf.nn.batch_normalization(fc1l, mean, variance, offset, scale, 1e-10)

                    fc1 = tf.nn.sigmoid(fc1l)
                with tf.variable_scope('fc2'):
                    self.weight_fc_2 = tf.get_variable('weight', [4096, 4096])
                    self.biases_fc_2 = tf.get_variable('biases', [4096])
                    fc12 = tf.nn.bias_add(tf.matmul(fc1, self.weight_fc_2), self.biases_fc_2)
                    fc2 = tf.nn.sigmoid(fc12)
                with tf.variable_scope('fc3'):
                    self.weight_fc_3 = tf.get_variable('weight', [4096, cfg.FLAGS.imagenet_out_number])
                    self.biases_fc_3 = tf.get_variable('biases', [cfg.FLAGS.imagenet_out_number])
                    fc3 = tf.nn.bias_add(tf.matmul(fc2, self.weight_fc_3), self.biases_fc_3)
                    # fc3 = tf.nn.relu(fc13)
        self.reuse = True
        # self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg')
        return fc3

    def train_loss(self, fc_out, labels):
        prediction = tf.nn.softmax(fc_out)
        cross_entropy = -tf.reduce_sum(labels * tf.log(prediction))        # 求和
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return train_step, accuracy
