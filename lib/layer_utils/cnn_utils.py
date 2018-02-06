import tensorflow as tf
from lib.config.config import FLAGS
from tensorflow.python.training import moving_averages

def batch_norm(name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if FLAGS.data_name == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        FLAGS.extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        FLAGS.extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)
      # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

def train_loss(prediction, labels, optimizer_name="Mom"):
    # demo1
    # prediction = tf.nn.softmax(prediction)
    # cross_entropy = -tf.reduce_sum(labels * tf.log(prediction))        # 求和
    # demo2
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels)  # label:one hot label
    cross_entropy_loss = tf.reduce_sum(cross_entropy)/FLAGS.batch_size
    weight_decay_loss = _decay()
    loss = cross_entropy_loss + weight_decay_loss


    tf.summary.scalar(loss.op.name, loss)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(loss, trainable_variables)

    if optimizer_name == "Mom":  # Momentum
        optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9)
    elif optimizer_name == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
    elif optimizer_name == 'ADAM':
        optimizer = tf.train.AdamOptimizer(1e-4)
    else:        # SGD
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

    # 生成一个变量用于保存全局训练步骤( global training step )的数值,并使用 minimize() 函数更新系统中的三角权重( triangle weights )、增加全局步骤的操作
    # global_step定义存储训练次数，从1开始自己随着训练次数增加而增加。因此这个变量不需要训练的
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # train_step = optimizer.minimize(loss, global_step=global_step)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=global_step, name='train_step')

    train_ops = [apply_op] + FLAGS.extra_train_ops
    train_step = tf.group(*train_ops)
    # ----------------------------------------------------------------------------
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    top_5 = tf.nn.in_top_k(predictions=prediction, targets=tf.cast(tf.argmax(labels, 1), tf.int32), k=5)   # predictions: one hot

    top_5 = tf.reduce_max(tf.cast(top_5, tf.float16))
    tf.summary.scalar("Loss_Function", loss)
    tf.summary.scalar("acc", accuracy)
    return train_step, accuracy, top_5, loss


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    # TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer, trainable=True)
    return new_variables


def _decay():
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
        costs.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)
    return tf.multiply(FLAGS.weight_decay, tf.add_n(costs))


class lr:
    def __init__(self):
        self.lr = 0.01
        self.reuse = False
        self.max_acc = 0
        self.count = 0
        self.max_count = 4

    def learning_rate(self, iter, acc):
        if self.lr == 0.0001:
            self.max_count = 7
        if self.lr == 0.00001:
            self.max_count = 8
        if iter < 2000:
            self.lr = 0.001
        elif iter == 2000:   # 防止开始的时候数据过大，开始使用较小的学习速率 int lr =0.01 or 0.001
            self.lr = 0.1
            self.max_acc = acc
            self.count = 0
            self.reuse = True
        if self.reuse:
            if acc >= self.max_acc:
                self.max_acc = acc
                self.count = 0
            else:
                self.count += 1
                if self.count > self.max_count:
                    self.lr /= 10
                    self.count = 0
                else:
                    pass
        else:
            pass
        if self.lr < 0.000001:
            self.lr = 0.000001
        return self.lr


def learning_rate(iter, data='cifar10'):
    if data == 'cifar10':
        if iter < 1001:   # 防止开始的时候数据过大，开始使用较小的学习速率
            lr = 0.01
        elif 1000 < iter < 40001:
            lr = 0.1
        elif 40000 < iter < 68001:
            lr = 0.01
        elif 68000 < iter < 78001:
            lr = 0.001
        elif 78000 < iter < 90001:
            lr = 0.0001
        else:
            lr = 0.00001
        return lr
    elif data == 'cifar100':
        if iter < 401:  # 防止开始的时候数据过大，开始使用较小的学习速率
            lr = 0.01
        elif 400 < iter < 32001:
            lr = 0.1
        elif 32000 < iter < 48001:
            lr = 0.01
        elif 48000 < iter < 64001:
            lr = 0.001
        elif 64000 < iter < 80001:
            lr = 0.0001
        elif 80000 < iter < 90001:
            lr = 0.00005
        else:
            lr = 0.00001
        return lr
    elif data == 'ImageNet':
        if iter < 1001:   # 防止开始的时候数据过大，开始使用较小的学习速率
            lr = 0.001
        elif 1000 < iter < 100001:
            lr = 0.1
        elif 100000 < iter < 200001:
            lr = 0.01
        elif 200000 < iter < 300001:
            lr = 0.001
        elif 300000 < iter < 400001:
            lr = 0.0001
        elif 400000 < iter < 500001:
            lr = 0.00005
        else:
            lr = 0.00001
        return lr
