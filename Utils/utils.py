import numpy as np
import random
import tensorflow as tf


def myTFfftshift2(inp):
    """
    2D fftshift for tf tensor
    :param inp: 4D (BHWC) input tensor
    :return: 4D (BHWC) output tensor
    """
    hnY = int(int(inp.shape[1]) / 2)
    hnX = int(int(inp.shape[2]) / 2)
    out = tf.concat([tf.slice(inp, [0, hnY, 0, 0], [-1, hnY, -1, -1]), tf.slice(inp, [0, 0, 0, 0], [-1, hnY, -1, -1])],
                    axis=1)
    out = tf.concat([tf.slice(out, [0, 0, hnX, 0], [-1, -1, hnX, -1]), tf.slice(out, [0, 0, 0, 0], [-1, -1, hnX, -1])],
                    axis=2)

    return out


def myTFfft2(inp):
    """
    2D fft for tf tensor
    :param inp: 4D (BHWC) input tensor
    :return: 4D (BHWC) output tensor
    """
    return myTFfftshift2(tf.transpose(tf.fft2d(tf.transpose(tf.cast(inp, tf.complex64), [0, 3, 1, 2])), [0, 2, 3, 1]))


def myTFifft2(inp):
    """
    2D ifft for tf tensor
    :param inp: 4D (BHWC) input tensor
    :return: 4D (BHWC) output tensor
    """
    return tf.transpose(tf.ifft2d(tf.transpose(myTFfftshift2(tf.cast(inp, tf.complex64)), [0, 3, 1, 2])), [0, 2, 3, 1])


def tf_ri2complex(img_ri):
    """
    real/imaginary to complex (tf version)
    :param img_ri: 4D (BHWC) input tensor
    :return: 4D (BHWC) output tensor
    """
    hnOut = int(int(img_ri.shape[3]) / 2)
    i_real = tf.slice(img_ri, [0, 0, 0, 0], [-1, -1, -1, hnOut])
    i_imag = tf.slice(img_ri, [0, 0, 0, hnOut], [-1, -1, -1, hnOut])
    comp = tf.complex(i_real, i_imag)
    return comp


def tf_complex2ri(inp):
    """
    complex to real/imaginary (tf version)
    :param inp: 4D (BHWC) input tensor
    :return: 4D (BHWC) output tensor
    """
    real = tf.cast(tf.real(inp), tf.float32)
    imag = tf.cast(tf.imag(inp), tf.float32)
    return tf.concat([real, imag], axis=3)


def tf_ri2ssos(img_ri):
    """
    real/imaginary to square root of sum of the squares (tf version)
    :param img_ri: 4D (BHWC) input tensor
    :return: 4D (BHWC) output tensor
    """
    hnOut = int(int(img_ri.shape[3]) / 2)
    i_real = tf.slice(img_ri, [0, 0, 0, 0], [-1, -1, -1, hnOut])
    i_imag = tf.slice(img_ri, [0, 0, 0, hnOut], [-1, -1, -1, hnOut])
    i_ssos = tf.sqrt(tf.reduce_sum(tf.square(i_real) + tf.square(i_imag), axis=3, keep_dims=True))
    return i_ssos


def ri2complex(img_ri):
    """
    real/imaginary to complex (np version)
    :param img_ri: 3D (HWC) input numpy array
    :return: 3D (HWC) output numpy array
    """
    hnOut = int(int(img_ri.shape[2]) / 2)
    i_real = img_ri[:, :, :hnOut]
    i_imag = img_ri[:, :, hnOut:]
    comp = i_real + 1j * i_imag
    return comp


def complex2ri(inp):
    """
    complex to real/imaginary (np version)
    :param inp: 3D (HWC) input numpy array
    :return: 3D (HWC) output numpy array
    """
    real = np.real(inp)
    imag = np.imag(inp)
    return np.concatenate([real, imag], axis=2)


def ri2ssos(img_ri):
    """
    real/imaginary to square root of sum of the squares (np version)
    :param img_ri: 3D (HWC) input numpy array
    :return: 3D (HWC) output numpy array
    """
    hnOut = int(int(img_ri.shape[2]) / 2)
    i_real = img_ri[:, :, :hnOut]
    i_imag = img_ri[:, :, hnOut:]
    i_ssos = np.sqrt(np.sum(i_real ** 2 + i_imag ** 2, axis=2, keepdims=True))
    return i_ssos


def fft2c(inp):
    """
    fft2 for multi-coil image
    :param inp: 3D (HWC) input numpy array
    :return: 3D (HWC) output numpy array
    """
    nC = inp.shape[2]
    out = np.zeros_like(inp)
    for c in range(nC):
        out[:, :, c] = np.fft.fft2(np.squeeze(inp[:, :, c]))
    return out


def ifft2c(inp):
    """
    ifft2 for multi-coil image
    :param inp: 3D (HWC) input numpy array
    :return: 3D (HWC) output numpy array
    """
    nC = inp.shape[2]
    out = np.zeros_like(inp)
    for c in range(nC):
        out[:, :, c] = np.fft.ifft2(np.squeeze(inp[:, :, c]))
    return out


def convert2int(img):
    """
    Convert float type image to uint8 type image for tensorboard writing
    :param img: 4D (BHWC) input tensor
    :return: 4D (BHWC) output tensor
    """
    img_max = tf.reduce_max(img, axis=[1, 2, 3], keepdims=True)
    img_min = tf.reduce_min(img, axis=[1, 2, 3], keepdims=True)
    int_img = tf.cast(255.0 * (img - img_min) / (img_max - img_min), tf.uint8)
    return int_img


def myNumExt(s):
    """
    Find the number in the file name
    :param s: string, file name
    :return: integer, the number in the file name
    """
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    return int(tail)


def get_lr(init_lr, num_epoch, epoch, decay_epoch):
    """
    Learning rate scheduler (linear decay)
    :param init_lr: float, initial learning rate
    :param num_epoch: integer, the number of epochs
    :param epoch: integer, current epoch
    :param decay_epoch: integer, epoch to start learning rate decay
    :return:
    """
    if epoch < decay_epoch:
        lr = init_lr
    else:
        lr = init_lr * (num_epoch - epoch) / (num_epoch - decay_epoch)
    return lr


class ImagePool:
    def __init__(self, pool_size):
        """
        Image pool for the discriminator of cycleGAN
        :param pool_size: integer, the size of the image pool
        """
        self.pool_size = pool_size
        self.images = []

    def __call__(self, image):
        if self.pool_size <= 0:
            return image

        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image
