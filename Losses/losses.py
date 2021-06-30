import tensorflow as tf


def LSGAN_loss(pred, target):
    return tf.reduce_mean(tf.squared_difference(pred, target))


def cycle_consistency_loss(real, recon):
    return tf.reduce_mean(tf.abs(recon - real))
