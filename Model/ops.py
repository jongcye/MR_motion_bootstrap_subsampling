import tensorflow as tf
import tensorflow.contrib.layers as li


def instance_norm(inp):
    with tf.variable_scope('instance_norm'):
        depth = inp.get_shape()[3]
        scale = tf.get_variable('scale', [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable('offset', [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(inp, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (inp - mean) * inv
        return scale * normalized + offset


# Generator layers
def ConvBlock(inp, f, reuse=False, name='ConvBlock'):
    """
    Convolution block of the generator
    :param inp: 4D (BHWC) input tensor
    :param f: integer, the number of filters
    :param reuse: boolean, true if reuse defined variable
    :param name: string
    :return: 4D (BHWC) output tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        padded = tf.pad(inp, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')

        W = tf.get_variable('W', shape=[3, 3, int(padded.get_shape()[3]), f], dtype=tf.float32,
                            initializer=li.xavier_initializer())
        b = tf.get_variable('b', shape=[f], dtype=tf.float32, initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(padded, W, strides=[1, 1, 1, 1], padding='VALID') + b

        normalized = instance_norm(conv)
        out = tf.nn.leaky_relu(normalized)
        return out


def MaxPool(inp, name='MaxPool'):
    """
    Max pooling layer of the generator
    :param inp: 4D (BHWC) input tensor
    :param name: string
    :return: 4D (BHWC) output tensor
    """
    with tf.variable_scope(name):
        out = tf.nn.max_pool(inp, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        return out


def UpBlock(inp, f, reuse=False, name='UpBlock'):
    """
    Unpooling block of the generator
    :param inp: 4D (BHWC) input tensor
    :param f: integer, the number of filters
    :param reuse: boolean, true if reuse defined variable
    :param name: string
    :return: 4D (BHWC) output tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        up = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(inp)
        padded = tf.pad(up, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')

        W = tf.get_variable('W', shape=[3, 3, int(padded.get_shape()[3]), f], dtype=tf.float32,
                            initializer=li.xavier_initializer())
        b = tf.get_variable('b', shape=[f], dtype=tf.float32, initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(padded, W, strides=[1, 1, 1, 1], padding='VALID') + b

        normalized = instance_norm(conv)
        out = tf.nn.leaky_relu(normalized)
        return out


def CC(down, up, name='CC'):
    """
    Concatenate layer of the generator
    :param down: 4D (BHWC) input tensor
    :param up: 4D (BHWC) input tensor
    :param name: string
    :return: 4D (BHWC) output tensor
    """
    with tf.variable_scope(name):
        out = tf.concat([down, up], axis=3)
        return out


def Conv1x1(inp, f, reuse=False, name='Conv1x1'):
    """
    Last 1x1 convolution layer of the generator
    :param inp: 4D (BHWC) input tensor
    :param f: integer, the number of filters
    :param reuse: boolean, true if reuse defined variable
    :param name: string
    :return: 4D (BHWC) output tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable('W', shape=[1, 1, int(inp.get_shape()[3]), f], dtype=tf.float32,
                            initializer=li.xavier_initializer())
        b = tf.get_variable('b', shape=[f], dtype=tf.float32, initializer=tf.zeros_initializer())
        out = tf.nn.conv2d(inp, W, strides=[1, 1, 1, 1], padding='VALID') + b
        return out


# Discriminator layers
def Ck(inp, f, s=2, norm='instance', reuse=False, name='Ck'):
    """
    4x4 convolution, nomalization, leakyReLU layers of the discriminator
    :param inp: 4D (BHWC) input tensor
    :param f: integer, the number of filters
    :param s: integer, stride
    :param reuse: boolean, true if reuse defined variable
    :param name: string
    :return: 4D (BHWC) output tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        padded = tf.pad(inp, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')

        W = tf.get_variable('W', shape=[4, 4, int(padded.get_shape()[3]), f], dtype=tf.float32,
                            initializer=li.xavier_initializer())
        b = tf.get_variable('b', shape=[f], dtype=tf.float32, initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(padded, W, strides=[1, s, s, 1], padding='VALID') + b

        if norm == 'instance':
            normalized = instance_norm(conv)
        else:
            normalized = conv
        out = tf.nn.leaky_relu(normalized)
        return out
