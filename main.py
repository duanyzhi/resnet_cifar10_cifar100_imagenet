import tensorflow as tf
import argparse

from lib.network.network import network
from lib.datasets.draw_img import plot_learning_curves
import lib.config.config as cfg

# 是否要解压1000个imagenet文件夹
# from lib.datasets.data import read_tar
# read_tar()


def run(pattern, data_name):
    net = network(pattern, data_name)
    net.resnet_model()
    sess = net.gpu_config()
    if pattern == 'train':
        test_acc_top1, train_acc_top1, loss = net.run_resnet(sess)
        plot_learning_curves(cfg.FLAGS.fig_path + data_name + '_resnet.png', len(test_acc_top1),
                             [test_acc_top1, train_acc_top1],
                             ['test', 'train'],  'ResNet - ' + data_name)
        plot_learning_curves(cfg.FLAGS.fig_path + data_name + '_resnet_loss.png', len(loss), [loss], ['loss'], 'Loss')
    else:
        net.run_resnet(sess)
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pattern', type=str, default='test',  # pattern前面加两个--表示pattern是可选参数
                        required=True, help='Choice train or test model')
    parser.add_argument('--data', type=str, default='cifar10',
                        required=True, help='Choice which dataset')

    args = parser.parse_args()
    print("Run ResNet with " + args.data + " for " + args.pattern)
    assert args.data == 'cifar10' or args.data == 'cifar100' or args.data == 'ImageNet'
    run(args.pattern, args.data)


# python main.py --pattern train --data cifar10
# python main.py --pattern test --data cifar10

# python main.py --pattern train --data cifar100
# python main.py --pattern test --data cifar100

# python main.py --pattern train --data ImageNet
# python main.py --pattern test --data ImageNet　
