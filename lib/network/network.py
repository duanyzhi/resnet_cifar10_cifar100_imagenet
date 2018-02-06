from lib.network.resnet_cifar import ResNet_cifar
from lib.network.resnet_imagenet import ResNet_ImageNet
from lib.layer_utils.cnn_utils import train_loss, lr
from lib.datasets.data import cifar, ImageNet
from lib.config.config import FLAGS
from DeepLearning.Image import plot_images
from DeepLearning.deep_learning import one_hot_to_index
from DeepLearning.python import list_save

import tensorflow as tf
import math

lr = lr()

class network:
    def __init__(self, pattern,  data_name):
        self.pattern = pattern
        self.data_name = data_name
        if data_name == 'cifar10' or data_name == 'cifar100':
            if data_name == 'cifar10':
                FLAGS.classes_number = 10
                FLAGS.depth_filter = 1
                FLAGS.iteration_numbers = 100001
            else:
                FLAGS.classes_number = 100
                FLAGS.depth_filter = 4
                FLAGS.iteration_numbers = 100001
            self.resnet = ResNet_cifar()
            self.data = cifar(data=data_name)
            self.x_placeholder = [None, FLAGS.cifar_img_size[0], FLAGS.cifar_img_size[1], FLAGS.cifar_img_size[2]]
            self.y_placeholder = [None, FLAGS.classes_number]
        elif data_name == 'ImageNet':
            FLAGS.classes_number = 1000
            FLAGS.batch_size = 32
            FLAGS.test_image_num = 50000
            FLAGS.iteration_numbers = 2000001
            FLAGS.saver_step = 100000
            self.x_placeholder = [None, FLAGS.imagenet_img_size[0], FLAGS.imagenet_img_size[1], FLAGS.imagenet_img_size[2]]
            self.y_placeholder = [None, FLAGS.classes_number]
            self.resnet = ResNet_ImageNet()
            self.data = ImageNet()
        self.placeholder()
        print("FLAGS.batch_size", FLAGS.batch_size)
        print("FLAGS.iteration_numbers", FLAGS.iteration_numbers)

    def placeholder(self):
        self.x = tf.placeholder(tf.float32, self.x_placeholder)
        self.y = tf.placeholder(tf.int32, self.y_placeholder)  # one hot的类型在int32或int64

    def resnet_model(self):
        self.fc_out = self.resnet(self.x, scope='resnet_' + self.data_name)
        self.train_step, self.accuracy, self.top_5, self.loss = train_loss(self.fc_out, self.y)
        self.prediction = tf.nn.softmax(self.fc_out)
        return

    def gpu_config(self):
        self.saver = tf.train.Saver()

        # 下面设置GPU分配方式可以有效避免出现Resource exhausted: OOM when allocating tensor with...等内存不足的情况
        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)  # 最多占用GPU 70%资源
        #  开始不会给tensorflow全部gpu资源 而是按需增加
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        if self.pattern == 'train':    # 只有训练模式才对变量初始化
            sess.run(tf.global_variables_initializer())
        return sess

    def run_resnet(self, sess):
        test_acc_top1 = [0]
        test_acc_top5 = [0]
        train_acc_top1 = [0]
        train_acc_top5 = [0]
        loss = []
        if self.pattern == 'train':
            for kk in range(1, FLAGS.iteration_numbers):
                img_batch, label_batch, _ = self.data(data_name="train")
                FLAGS.data_name = 'train'
                _, top_1, top_5, loss_ = sess.run([self.train_step, self.accuracy, self.top_5, self.loss],
                                                      feed_dict={self.x: img_batch, self.y: label_batch})
                if kk == 1:
                    loss.append(loss_)
                # show test acc
                if kk % FLAGS.display_step == 0:
                    te_acc_top1,  te_acc_top5 = self.run_test_data(sess)
                    FLAGS.learning_rate = lr.learning_rate(kk, te_acc_top1)
                    print("learning rate", FLAGS.learning_rate)
                    print("Resnet Model run " + self.data_name + ", the iter number is : %5d, the acc is : %.6f,　the loss is : %.6f "
                                                                  % (kk, te_acc_top1, loss_))
                    test_acc_top1.append(te_acc_top1)
                    # test_acc_top5.append(te_acc_top5)
                    train_acc_top1.append(top_1)
                    # train_acc_top5.append(top_5)
                    loss.append(loss_)
                if kk % FLAGS.saver_step == 0:
                    self.saver.save(sess, "data/ckpt/" + self.data_name + "_layers"+str(FLAGS.layers_depth) + '_' + str(kk) + "_model.ckpt")
                    list_save(test_acc_top1, 'data/'+self.data_name + '_test_top1.txt')
                    # list_save(test_acc_top5, 'data/' + self.data_name + '_test_top5.txt')
                    list_save(train_acc_top1, 'data/'+self.data_name + '_train_top1.txt')
                    # list_save(train_acc_top5, 'data/' + self.data_name + '_train_top5.txt')
            return test_acc_top1, train_acc_top1, loss
        else:  # test
            self.saver.restore(sess, "data/ckpt/" + self.data_name + "/" + self.data_name + "_layers"+str(FLAGS.layers_depth) + "_model.ckpt")
            # FLAGS.batch_size = 16  # 测试用的batch size小一点
            te_acc_top1, te_acc_top5 = self.run_test_data(sess)
            print("Test dataset acc: top1: %.6f, top5: %.6f" % (te_acc_top1, te_acc_top5))
            for jj in range(FLAGS.test_iter_num):      # 只输出5个batch
                img_batch, label_batch, img_real = self.data(data_name="test")
                fc_label = sess.run(self.prediction, feed_dict={self.x: img_batch, self.y: label_batch})
                pre_label = one_hot_to_index(fc_label)
                rel_label = one_hot_to_index(label_batch)
                print("pre_label: ", pre_label)
                print("rel_label: ", rel_label)
                plot_images(img_real, pre_label)
            return

    def run_test_data(self, sess):
        acc_test_top1 = 0
        acc_test_top5 = 0
        number_test = int(math.ceil(FLAGS.test_image_num / FLAGS.batch_size))  # 向上取整
        for jj in range(number_test):
            img_batch, label_batch, _ = self.data(data_name="test")
            FLAGS.data_name = 'test'
            test_top1, test_top5 = sess.run([self.accuracy, self.top_5], feed_dict={self.x: img_batch, self.y: label_batch})
            acc_test_top1 += test_top1
            acc_test_top5 += test_top5
        return acc_test_top1 / number_test, acc_test_top5 / number_test
