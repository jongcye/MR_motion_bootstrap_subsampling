import tensorflow as tf
from Model.ops import Ck


class PatchGAN:
    def __init__(self, opt, name):
        """
        PatchGAN discriminator
        :param opt: options
        :param name: string
        """
        self.ndf = opt.ndf  # the number of filters of the first layer
        self.name = name
        self.reuse = False

    def __call__(self, inp):
        with tf.variable_scope(self.name):
            C1 = Ck(inp, self.ndf, s=2, norm='none', reuse=self.reuse, name='C{}'.format(self.ndf))
            C2 = Ck(C1, self.ndf * 2, s=2, norm='instance', reuse=self.reuse, name='C{}'.format(self.ndf * 2))
            C3 = Ck(C2, self.ndf * 4, s=2, norm='instance', reuse=self.reuse, name='C{}'.format(self.ndf * 4))
            C4 = Ck(C3, self.ndf * 8, s=1, norm='instance', reuse=self.reuse, name='C{}'.format(self.ndf * 8))
            out = Ck(C4, 1, s=1, norm='none', reuse=self.reuse, name='out')

            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return out
