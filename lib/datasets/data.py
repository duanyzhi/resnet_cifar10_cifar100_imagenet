import struct
import random
import tarfile
import cv2
import os

import pickle as p
import numpy as np

import matplotlib.pyplot as plt
import lib.config.config as cfg

from DeepLearning.deep_learning import Batch_Normalization, one_hot
from DeepLearning.Image import plot_images

# -------------------------  mnist -------------------------


def load_CIFAR10_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = p.load(f, encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    return X, Y


def load_CIFAR100_batch(filename, num):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = p.load(f, encoding='latin1')
    X = datadict['data']
    fine = datadict['fine_labels']
    coarse = datadict['coarse_labels']
    X = X.reshape(num, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    fine = np.array(fine)
    coarse = np.array(coarse)
    return X, fine, coarse

def loadImageSet(binfile):
    buffers = binfile.read()
    head = struct.unpack_from('>IIII', buffers, 0)
    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'  # like '>47040000B'
    imgs = struct.unpack_from(bitsString, buffers, offset)
    binfile.close()
    imgs = np.reshape(imgs, [imgNum, 1, width * height])
    return imgs


def loadLabelSet(binfile):
    buffers = binfile.read()
    head = struct.unpack_from('>II', buffers, 0)
    imgNum = head[1]
    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])
    return labels


class mnist:
    def __init__(self):
        self.train_images_in = open("G:\\MNIST\\MNIST_data\\train-images.idx3-ubyte", 'rb')
        self.train_labels_in = open("G:\\MNIST\\MNIST_data\\train-labels.idx1-ubyte", 'rb')
        self.test_images_in = open("G:\\MNIST\\MNIST_data\\t10k-images.idx3-ubyte", 'rb')
        self.test_labels_in = open("G:\\MNIST\\MNIST_data\\t10k-labels.idx1-ubyte", 'rb')
        self.batch_size = cfg.FLAGS.batch_size
        self.train_image = loadImageSet(self.train_images_in)  # [60000, 1, 784]
        self.train_labels = loadLabelSet(self.train_labels_in)  # [60000, 1]
        self.test_images = loadImageSet(self.test_images_in)  # [10000, 1, 784]
        self.test_labels = loadLabelSet(self.test_labels_in)  # [10000, 1]
        self.data = {"train": self.train_image, "test": self.test_images}
        self.label = {"train": self.train_labels, "test": self.test_labels}
        self.indexes = {"train": 0, "val": 0, "test": 0}

    def get_mini_batch(self, data_name="train"):
        if (self.indexes[data_name] + 1) * self.batch_size > self.data[data_name].shape[0]:
            self.indexes[data_name] = 0
        batch_data = self.data[data_name][
                     self.indexes[data_name] * self.batch_size:(self.indexes[data_name] + 1) * self.batch_size, :, :]
        batch_label = self.label[data_name][
                      self.indexes[data_name] * self.batch_size:(self.indexes[data_name] + 1) * self.batch_size, :]
        self.indexes[data_name] += 1
        y = np.zeros((self.batch_size, len(cfg.FLAGS.chars)))
        for kk in range(self.batch_size):
            y[kk, cfg.FLAGS.chars.index(str(int(batch_label[kk])))] = 1.0
        x = Batch_Normalization(batch_data)
        x = np.reshape(x, (16, 784, 1))
        x = np.reshape(x, (16, 28, 28, 1))
        return x, y


# ---------------------  CIFAR10  ---------------------------------
class cifar:
    def __init__(self, data):
        if data == 'cifar10':
            x1, y1 = load_CIFAR10_batch(cfg.FLAGS.cifar10_data_path + "data_batch_1")  # 每一个data_batch是[10000, 32, 32, 3的大小]
            x2, y2 = load_CIFAR10_batch(cfg.FLAGS.cifar10_data_path + "data_batch_2")
            x3, y3 = load_CIFAR10_batch(cfg.FLAGS.cifar10_data_path + "data_batch_3")
            x4, y4 = load_CIFAR10_batch(cfg.FLAGS.cifar10_data_path + "data_batch_4")
            x5, y5 = load_CIFAR10_batch(cfg.FLAGS.cifar10_data_path + "data_batch_5")
            self.train_image = np.concatenate((x1, x2, x3, x4, x5), 0)  # (50000, 32, 32, 3) (50000,)
            self.train_labels = np.concatenate((y1, y2, y3, y4, y5))
            self.test_image, self.test_labels = load_CIFAR10_batch(cfg.FLAGS.cifar10_data_path + "test_batch")   # x, y: (10000, 32, 32, 3) (10000,)
        elif data == 'cifar100':
            self.train_image, self.train_labels, _ = load_CIFAR100_batch(
                cfg.FLAGS.cifar100_data_path + 'train', 50000)    # x, y: (50000, 32, 32, 3) (50000,)
            self.test_image, self.test_labels, _ = load_CIFAR100_batch(
                cfg.FLAGS.cifar100_data_path + 'test', 10000)    # x, y: (10000, 32, 32, 3) (10000,)
        else:
            print("The input name of cifar is not cifar10 or cifar100")
            raise NameError
        self.data = {"train": self.train_image, "test": self.test_image}
        self.label = {"train": self.train_labels, "test": self.test_labels}
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.batch_size = cfg.FLAGS.batch_size
        self.shuffle = True
        self.remain = {"train": 0, "val": 0, "test": 0}

    def __call__(self, data_name="train"):
        if (self.indexes[data_name] + 1) * self.batch_size + self.remain[data_name] > self.data[data_name].shape[0]:
            remain_num = self.data[data_name].shape[0] - self.indexes[data_name] * self.batch_size - self.remain[data_name]
            if remain_num != 0:
                batch_data_1 = self.data[data_name][-remain_num:, ...]
                batch_label_1 = self.label[data_name][-remain_num:]
                order = np.random.permutation(self.data[data_name].shape[0])
                self.data[data_name] = self.data[data_name][order, ...]
                self.label[data_name] = self.label[data_name][order]
                batch_data_2 = self.data[data_name][0:cfg.FLAGS.batch_size-remain_num, ...]
                batch_label_2 = self.label[data_name][0:cfg.FLAGS.batch_size-remain_num]
                batch_data = np.concatenate((batch_data_1, batch_data_2), 0)
                batch_label = np.concatenate((batch_label_1, batch_label_2))
                self.indexes[data_name] = -1
                self.remain[data_name] = cfg.FLAGS.batch_size - remain_num
            else:
                if self.shuffle is True:
                    # print('Shuffling')
                    order = np.random.permutation(self.data[data_name].shape[0])
                    self.data[data_name] = self.data[data_name][order, ...]
                    self.label[data_name] = self.label[data_name][order]
                self.indexes[data_name] = 0
                batch_data = self.data[data_name][
                             self.indexes[data_name] * self.batch_size:(self.indexes[data_name] + 1) * self.batch_size,
                             :, :, :]
                batch_label = self.label[data_name][
                              self.indexes[data_name] * self.batch_size:(self.indexes[data_name] + 1) * self.batch_size]
        else:
            batch_data = self.data[data_name][
                         self.indexes[data_name] * self.batch_size + self.remain[data_name]:(self.indexes[data_name] + 1) * self.batch_size + self.remain[data_name], :, :, :]
            batch_label = self.label[data_name][
                          self.indexes[data_name] * self.batch_size + self.remain[data_name]:(self.indexes[data_name] + 1) * self.batch_size + self.remain[data_name]]
        self.indexes[data_name] += 1
        y = one_hot(batch_label, cfg.FLAGS.classes_number)
        if data_name == "train":
            data_argument = np.zeros((cfg.FLAGS.batch_size, 32+4*2, 32+4*2, 3))
            data_argument[:, 4:36, 4:36, :] = batch_data
            rand_data = np.zeros_like(batch_data)
            for ii in range(cfg.FLAGS.batch_size):
                x_begin = np.random.randint(0, 8)
                y_begin = np.random.randint(0, 8)
                argument_img = data_argument[ii, x_begin:x_begin+32, y_begin:y_begin+32, :]
                # 50%可能性翻转  沿y轴水平旋转
                flip_prop = np.random.randint(low=0, high=2)
                if flip_prop == 0:
                    argument_img = cv2.flip(argument_img, 1)
                rand_data[ii, :, :, :] = argument_img
            batch_data = rand_data
        x = batch_data   # without bn
        x_bn = Batch_Normalization(batch_data)  # with bn
        return x_bn, y, x

# ---------------------  CIFAR100  ---------------------------------
"""
  cifar10 和cifar100的介绍网址：
        http://www.cs.toronto.edu/~kriz/cifar.html
  cifar10顺序: [airplane, automobile，bird，cat，deer, dog, frog, horse, ship, truck]
  cifar100:
  <1 x coarse label><1 x fine label><3072 x pixel>
  The  3072 bytes are the values of the pixels of the image. The first 1024(32*32) bytes are the red channel values,
       the next 1024 the green, and the final 1024 the blue.
  it has 100 classes containing 600 images each.
  There are 500 training images and 100 testing images per class.
  The 100 classes in the CIFAR-100 are grouped into 20 superclasses.
  Each image comes with a "fine" label (the class to which it belongs)
  and a "coarse" label (the superclass to which it belongs).
    Superclass	                    Classes
    aquatic mammals             	beaver, dolphin, otter, seal, whale
    fish	                        aquarium fish, flatfish, ray, shark, trout
    flowers	                        orchids, poppies, roses, sunflowers, tulips
    food containers	                bottles, bowls, cans, cups, plates
    fruit and vegetables            apples, mushrooms, oranges, pears, sweet peppers
    household electrical devices    clock, computer keyboard, lamp, telephone, television
    household furniture	            bed, chair, couch, table, wardrobe
    insects	                        bee, beetle, butterfly, caterpillar, cockroach
    large carnivores	            bear, leopard, lion, tiger, wolf
    large man-made outdoor things   bridge, castle, house, road, skyscraper
    large natural outdoor scenes    cloud, forest, mountain, plain, sea
    large omnivores and herbivores  camel, cattle, chimpanzee, elephant, kangaroo
    medium-sized mammals            fox, porcupine, possum, raccoon, skunk
    non-insect invertebrates        crab, lobster, snail, spider, worm
    people                      	baby, boy, girl, man, woman
    reptiles	                    crocodile, dinosaur, lizard, snake, turtle
    small mammals                   hamster, mouse, rabbit, shrew, squirrel
    trees                        	maple, oak, palm, pine, willow
    vehicles 1	                    bicycle, bus, motorcycle, pickup truck, train
    vehicles 2	                    lawn-mower, rocket, streetcar, tank, tractor
"""

# ------------------------- ImageNet -------------------------------------
# 解压imagenet文件夹
def read_tar():
    def untar(fname, dirs):  # dirs是文件提取出来后保存的目录
        t = tarfile.open(fname)
        t.extractall(path=dirs)
    for name in os.listdir(os.path.join(cfg.FLAGS.imagenet_data_path, 'train')):
        new_path = os.path.join(cfg.FLAGS.imagenet_data_path, 'train', name[:9])
        os.mkdir(new_path)
        untar(os.path.join(cfg.FLAGS.imagenet_data_path, 'train', name), new_path)

class ImageNet:
    def __init__(self):
        # Load Train ImageNet
        self.train_image = []
        for file in os.listdir(os.path.join(cfg.FLAGS.imagenet_data_path, 'JPEG')):
            for img_name in os.listdir(os.path.join(cfg.FLAGS.imagenet_data_path, 'JPEG', file)):
                self.train_image.append(img_name)
        # Load val ImageNet, no test imagenet label so we use val imagenet for test
        self.val_image = []
        for val_file in os.listdir(os.path.join(cfg.FLAGS.imagenet_data_path, 'ILSVRC2012_img_val')):
            self.val_image.append(val_file)
        random.shuffle(self.train_image)
        f = open(os.path.join(cfg.FLAGS.imagenet_data_path, 'label', 'ImageNet_label.txt'), "r")
        lines = f.readlines()
        f.close()
        self.label_list = []
        for index, f in enumerate(lines):
            self.label_list.append(f[:9])

        f = open(os.path.join(cfg.FLAGS.imagenet_data_path, 'label', 'ILSVRC2012_validation_ground_truth.txt'), "r")
        lines = f.readlines()
        f.close()
        self.test_label = []   # 注意test label是从1开始到1000不是从0到999的
        for line in lines:
            self.test_label.append(line[:-1])  # 最后一个是换行符不要

        self.data = {"train": self.train_image, "test": self.val_image}
        self.label = {"train": self.label_list, "test": self.test_label}
        self.indexes = {"train": 0, "test": 0}
        self.batch_size = cfg.FLAGS.batch_size
        self.shuffle = True

    def __call__(self, data_name="train"):
        if (self.indexes[data_name] + 1) * self.batch_size > len(self.data[data_name]):
            if self.shuffle is True and data_name == 'train':  # val image no shuffle
                random.shuffle(self.data[data_name])
            self.indexes[data_name] = 0
        batch_data = np.zeros((cfg.FLAGS.batch_size, cfg.FLAGS.imagenet_img_size[0], cfg.FLAGS.imagenet_img_size[1], cfg.FLAGS.imagenet_img_size[2]))
        batch_label = np.zeros((cfg.FLAGS.batch_size, cfg.FLAGS.classes_number))
        batch_list = self.data[data_name][self.indexes[data_name] * self.batch_size:(self.indexes[data_name]+1) * self.batch_size]

        for ind, file in enumerate(batch_list):
            if data_name == 'train':
                im = cv2.imread(os.path.join(cfg.FLAGS.imagenet_data_path,  'JPEG', file[:9], file))
                batch_label[ind, self.label[data_name].index(file[:9])] = 1.0
            else:
                im = cv2.imread(os.path.join(cfg.FLAGS.imagenet_data_path, 'ILSVRC2012_img_val', file))
                # print(file, eval(self.label[data_name][self.indexes[data_name] * self.batch_size + ind]))
                batch_label[ind, eval(self.label[data_name][self.indexes[data_name] * self.batch_size + ind]) - 1] = 1.0
            h, w, _ = im.shape  # (height, weight, depth)
            # print(im.shape)
            # plt.imshow(im)
            # plt.show()
            if h > w:
                im = cv2.resize(im, (256, 480))  # im.shape: [480, 256, 3] 480是高
                h_begin = 480 - 224
                w_begin = 256 - 224
                argument_img = im[h_begin:h_begin + 224, w_begin:w_begin + 224, :]
            else:
                im = cv2.resize(im, (480, 256))  # im.shape: [256, 480, 3] 480是宽
                h_begin = 256 - 224
                w_begin = 480 - 224
                argument_img = im[h_begin:h_begin + 224, w_begin:w_begin + 224, :]
            # 50%可能性翻转  沿y轴水平旋转
            flip_prop = np.random.randint(low=0, high=2)
            if flip_prop == 0:
                argument_img = cv2.flip(argument_img, 1)
            batch_data[ind, :, :, :] = argument_img

        self.indexes[data_name] += 1

        # plot_images(batch_data, batch_label)
        x = Batch_Normalization(batch_data)
        # plot_images(x, batch_label)
        return x, batch_label, batch_data




