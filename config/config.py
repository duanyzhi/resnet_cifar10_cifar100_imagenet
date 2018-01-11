__author__ = "Nova Future"
__date__ = "$2018-1-4"

"""
   ResNet -- cifar10
   ResNet -- cifar100
   ResNet -- ImageNet

"""
import os
import platform
import os.path as osp
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

#######################
# Training Parameters #
#######################
tf.app.flags.DEFINE_float('weight_decay', 0.0001, "Weight decay, for regularization")
tf.app.flags.DEFINE_float('epsilon', 1e-4, "epsilon for BN")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "Learning rate")
tf.app.flags.DEFINE_float('momentum', 0.9, "Momentum")
tf.app.flags.DEFINE_float('gamma', 0.1, "Factor for reducing the learning rate")

tf.app.flags.DEFINE_integer('batch_size', 128, "Network batch size:cifar10 :128; ImageNet:")
tf.app.flags.DEFINE_integer('iteration_numbers', 90001, "iteration number for train 多个1方便后面保存模型"
                                                         "cifar100 iter:80000"
                                                          "cifar10 iter:70000")
tf.app.flags.DEFINE_integer('test_iter_num', 5, "iteration number for test")

tf.app.flags.DEFINE_integer('saver_step', 10000,
                            "the step of save the model")
tf.app.flags.DEFINE_integer('display_step', 1000,
                            "Iteration intervals for showing the loss during training, on command line interface")
tf.app.flags.DEFINE_string('log_dir', "tensorboard\\", '''Path to save log and checkpoint''')  # string类型
tf.app.flags.DEFINE_integer('layers_depth', 110, """the depth of layers about resnet
                                            cifar10: 20 32 56 110 
                                            cifar10: 20 32 56 110 
                                            cifar100: 110
                                            ImageNet: 50 101 152""")

#######################
#     ResNet Model    #
#######################
tf.app.flags.DEFINE_integer('imagenet_img_size', [224, 224, 3], "The shape of ImageNet img input to resnet")
tf.app.flags.DEFINE_integer('cifar_img_size', [32, 32, 3], "The shape of cifar img input to resnet")

tf.app.flags.DEFINE_integer('classes_number', 10, '''classes number cifar10 or cifar100 and imagenet, defaule is 10''')
tf.app.flags.DEFINE_integer('depth_filter', 1, """depth of cifar resnet :k default is 1, Ref:Wide-ResNet""")
#######################
#       Data Set      #
#######################
tf.app.flags.DEFINE_string('fig_path', 'data/fig/', "fig path of acc and iters")
tf.app.flags.DEFINE_integer('color', ['dodgerblue', 'red', 'aqua', 'orange'], 'color for plt.plot')

if platform.system() == 'Windows':  # windows:
    tf.app.flags.DEFINE_string('cifar10_data_path', 'E:\\cifar10\\', "path of cifar10")
    tf.app.flags.DEFINE_string('cifar100_data_path', 'E:\\cifar100\\', "path of cifar100")
    tf.app.flags.DEFINE_string('imagenet_data_path', 'E:\\ImageNet\\', "path of ImageNet")
else:  # ubuntu
    tf.app.flags.DEFINE_string('cifar10_data_path', '/media/dyz/Data/cifar10/', "path of cifar10")
    tf.app.flags.DEFINE_string('cifar100_data_path', '/media/dyz/Data/cifar100/', "path of cifar100")
    tf.app.flags.DEFINE_string('imagenet_data_path', '/media/dyz/Data/ImageNet', "path of ImageNet")
