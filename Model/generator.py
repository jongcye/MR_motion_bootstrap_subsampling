import tensorflow as tf
from Utils.utils import myTFfft2, myTFifft2, tf_ri2complex, tf_complex2ri
from Model.ops import ConvBlock, MaxPool, UpBlock, CC, Conv1x1


class Unet:
    def __init__(self, opt, name):
        """
        U-Net generator
        :param opt: options
        :param name: string
        """
        self.ngf = opt.ngf  # the number of filters of the first layer
        self.last_nc = opt.nC  # 1 for the magnitude image, 2 * the number of coils for complex image
        self.n = 5
        self.name = name
        self.reuse = False

    def __call__(self, inp):
        with tf.variable_scope(self.name):
            tmp = tf.identity(inp)
            d_list = []
            for n in range(self.n - 1):
                CBR1 = ConvBlock(tmp, self.ngf * (2 ** n), reuse=self.reuse, name='CBR_{}_down1'.format(self.ngf * (2 ** n)))
                CBR2 = ConvBlock(CBR1, self.ngf * (2 ** n), reuse=self.reuse, name='CBR_{}_down2'.format(self.ngf * (2 ** n)))
                pool = MaxPool(CBR2, name='Pool_{}'.format(self.ngf * (2 ** n)))
                d_list.append(CBR2)
                tmp = pool

            CBR1 = ConvBlock(tmp, self.ngf * (2 ** (self.n - 1)), reuse=self.reuse, name='CBR_{}_1'.format(self.ngf * (2 ** n)))
            CBR2 = ConvBlock(CBR1, self.ngf * (2 ** (self.n - 1)), reuse=self.reuse, name='CBR_{}_2'.format(self.ngf * (2 ** n)))

            tmp = CBR2
            for n in range(self.n - 1):
                up = UpBlock(tmp, self.ngf * (2 ** (self.n - n - 2)), reuse=self.reuse, name='Up_{}'.format(self.ngf * (2 ** (self.n - n - 2))))
                cc = CC(d_list[self.n - n - 2], up, name='CC_{}'.format(self.ngf * (2 ** (self.n - n - 2))))
                CBR1 = ConvBlock(cc, self.ngf * (2 ** (self.n - n - 2)), reuse=self.reuse,name='CBR_{}_up1'.format(self.ngf * (2 ** (self.n - n - 2))))
                CBR2 = ConvBlock(CBR1, self.ngf * (2 ** (self.n - n - 2)), reuse=self.reuse, name='CBR_{}_up2'.format(self.ngf * (2 ** (self.n - n - 2))))
                tmp = CBR2

            c1x1 = Conv1x1(tmp, self.last_nc, reuse=self.reuse, name='Conv1x1')

            res = c1x1 + inp
            cc = CC(c1x1, res)
            out = Conv1x1(cc, self.last_nc, reuse=self.reuse, name='out')

            if self.last_nc == 1:
                out = tf.nn.relu(out)  # magnitude image -> only positive values

            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return out


class Downsampling:
    def __init__(self, name):
        """
        Fourier transform -> subsampling -> inverse Fourier transform
        :param name: string
        """
        self.name = name

    def __call__(self, inp, mask):
        with tf.variable_scope(self.name):
            if int(inp.get_shape()[3]) == 1:
                full_k = myTFfft2(inp)  # magnitude image
            else:
                full_k = myTFfft2(tf_ri2complex(inp))  # complex image
            down_k = tf.multiply(full_k, tf.cast(mask, tf.complex64))
            if int(inp.get_shape()[3]) == 1:
                out = tf.abs(myTFifft2(down_k))
            else:
                out = tf_complex2ri(myTFifft2(down_k))

            return out
