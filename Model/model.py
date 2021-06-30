import numpy as np
import tensorflow as tf
from math import ceil
from os import makedirs
from os.path import join, isdir
from scipy import io as sio
from tqdm import tqdm
from Losses.losses import LSGAN_loss, cycle_consistency_loss
from Model.generator import Unet, Downsampling
from Model.discriminator import PatchGAN
from Utils.utils import tf_ri2ssos, convert2int, myNumExt, get_lr, ImagePool


class MR_motion_magnitude:
    def __init__(self, sess, opt):
        """
        Model for MR motion reduction using bootstrap subsampling and aggregation (for magnitude image)
        :param sess: tf session
        :param opt: options
        """
        GPU = '/device:GPU:' + str(opt.gpu_ids[0])
        print(GPU)

        self.GPU = GPU
        self.sess = sess
        self.disp_div_N = opt.disp_div_N
        self.batch_size = opt.batch_size
        self.nY = opt.nY
        self.nX = opt.nX
        self.nC = opt.nC
        self.lambda_cycle = opt.lambda_cycle
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.num_epoch = opt.num_epoch
        self.decay_epoch = opt.decay_epoch
        self.N = opt.N
        self.init_lr = opt.init_lr
        self.save_epoch = opt.save_epoch
        self.save_path = opt.save_path
        self.experiment_name = opt.experiment_name
        self.opt = opt

        self.num_step_train = opt.num_step_train
        self.num_step_test = opt.num_step_test
        self.ckpt_dir = opt.ckpt_dir

        self._build_model()
        self.saver = tf.train.Saver(name='saver')
        self.writer = tf.summary.FileWriter(opt.log_dir, self.sess.graph)
        self.pool = ImagePool(opt.pool_size)

    def _build_model(self):
        with tf.device(self.GPU):
            self.G_D2F = Unet(self.opt, name='Gen_D2F')
            self.G_F2D = Downsampling(name='Gen_F2D')
            self.D_F = PatchGAN(self.opt, name='Dis_F')

            self.real_F = tf.placeholder(tf.float32, [None, self.nY, self.nX, self.nC], name='real_F')
            self.real_D = tf.placeholder(tf.float32, [None, self.nY, self.nX, self.nC], name='real_D')
            self.mask_F = tf.placeholder(tf.float32, [None, self.nY, self.nX, self.nC], name='mask_F')
            self.mask_D = tf.placeholder(tf.float32, [None, self.nY, self.nX, self.nC], name='mask_D')

            self.fake_F = self.G_D2F(self.real_D)
            self.fake_D = self.G_F2D(self.real_F, self.mask_F)

            self.recon_F = self.G_D2F(self.fake_D)
            self.recon_D = self.G_F2D(self.fake_F, self.mask_D)

            self.dis_real_F = self.D_F(self.real_F)
            self.dis_fake_F = self.D_F(self.fake_F)

            self.G_fake_gan_loss = LSGAN_loss(self.dis_fake_F, tf.ones_like(self.dis_fake_F))

            self.cycle_F_loss = cycle_consistency_loss(self.real_F, self.recon_F)
            self.cycle_D_loss = cycle_consistency_loss(self.real_D, self.recon_D)

            self.G_total_loss = self.G_fake_gan_loss + self.lambda_cycle * (self.cycle_F_loss + self.cycle_D_loss)

            self.fake_F_pool = tf.placeholder(tf.float32, [None, self.nY, self.nX, self.nC], name='fake_F_pool')
            self.dis_fake_F_pool = self.D_F(self.fake_F_pool)

            self.D_real_gan_loss = LSGAN_loss(self.dis_real_F, tf.ones_like(self.dis_real_F))
            self.D_fake_gan_loss = LSGAN_loss(self.dis_fake_F_pool, tf.zeros_like(self.dis_fake_F_pool))

            self.D_total_loss = (self.D_real_gan_loss + self.D_fake_gan_loss) / 2

            self.G_fake_gan_sum = tf.summary.scalar('G/1_fake_gan_loss', self.G_fake_gan_loss)
            self.cycle_F_sum = tf.summary.scalar('G/2_cycle_F_loss', self.cycle_F_loss)
            self.cycle_D_sum = tf.summary.scalar('G/2_cycle_D_loss', self.cycle_D_loss)
            self.G_total_sum = tf.summary.scalar('G/3_total_loss', self.G_total_loss)
            self.G_sum = tf.summary.merge([self.G_fake_gan_sum, self.cycle_F_sum, self.cycle_D_sum, self.G_total_sum])

            self.D_real_gan_sum = tf.summary.scalar('D/1_real_gan_loss', self.D_real_gan_loss)
            self.D_fake_gan_sum = tf.summary.scalar('D/1_fake_gan_loss', self.D_fake_gan_loss)
            self.D_total_sum = tf.summary.scalar('D/2_total_loss', self.D_total_loss)
            self.D_sum = tf.summary.merge([self.D_real_gan_sum, self.D_fake_gan_sum, self.D_total_sum])

            self.real_F_sum = tf.summary.image('FDF/1_real_F', convert2int(self.real_F), max_outputs=1)
            self.real_D_sum = tf.summary.image('DFD/1_real_D', convert2int(self.real_D), max_outputs=1)
            self.fake_F_sum = tf.summary.image('DFD/1_fake_F', convert2int(self.fake_F), max_outputs=1)
            self.fake_D_sum = tf.summary.image('FDF/1_fake_D', convert2int(self.fake_D), max_outputs=1)
            self.recon_F_sum = tf.summary.image('FDF/3_recon_F', convert2int(self.recon_F), max_outputs=1)
            self.recon_D_sum = tf.summary.image('DFD/3_recon_D', convert2int(self.recon_D), max_outputs=1)
            self.img_sum = tf.summary.merge([self.real_F_sum, self.real_D_sum, self.fake_F_sum, self.fake_D_sum, self.recon_F_sum, self.recon_D_sum])

            self.lr = tf.placeholder(tf.float32, None, name='lr')
            self.G_fake_gan_avg_loss = tf.placeholder(tf.float32, None, name='G_fake_gan_avg_loss')
            self.cycle_F_avg_loss = tf.placeholder(tf.float32, None, name='cycle_F_avg_loss')
            self.cycle_D_avg_loss = tf.placeholder(tf.float32, None, name='cycle_D_avg_loss')
            self.D_real_gan_avg_loss = tf.placeholder(tf.float32, None, name='D_real_gan_avg_loss')
            self.D_fake_gan_avg_loss = tf.placeholder(tf.float32, None, name='D_fake_gan_avg_loss')

            self.lr_sum = tf.summary.scalar('Epoch/1_lr', self.lr)
            self.G_fake_gan_avg_sum = tf.summary.scalar('Epoch/2_G_fake_gan_avg_loss', self.G_fake_gan_avg_loss)
            self.cycle_F_avg_sum = tf.summary.scalar('Epoch/3_cycle_F_avg_loss', self.cycle_F_avg_loss)
            self.cycle_D_avg_sum = tf.summary.scalar('Epoch/3_cycle_D_avg_loss', self.cycle_D_avg_loss)
            self.D_real_gan_avg_sum = tf.summary.scalar('Epoch/4_D_real_gan_avg_loss', self.D_real_gan_avg_loss)
            self.D_fake_gan_avg_sum = tf.summary.scalar('Epoch/4_D_fake_gan_avg_loss', self.D_fake_gan_avg_loss)
            self.epoch_sum = tf.summary.merge([self.lr_sum, self.G_fake_gan_avg_sum, self.cycle_F_avg_sum, self.cycle_D_avg_sum, self.D_real_gan_avg_sum, self.D_fake_gan_avg_sum])

            self.G_D2F_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2).minimize(self.G_total_loss, var_list=self.G_D2F.variables)
            self.D_F_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2).minimize(self.D_total_loss, var_list=self.D_F.variables)

    def train(self, dataloader):
        disp_step = ceil(self.num_step_train / self.disp_div_N)

        latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        if latest_ckpt is None:
            print('Start with random initialization')
            self.sess.run(tf.global_variables_initializer())
            epoch_start = 0
        else:
            print('Start from saved model - ' + latest_ckpt)
            self.saver.restore(self.sess, latest_ckpt)
            epoch_start = myNumExt(latest_ckpt)

        for epoch in tqdm(range(epoch_start, self.num_epoch), desc='Epoch', total=self.num_epoch, initial=epoch_start):
            disp_cnt = 0

            dataloader.shuffle(domain='F', seed=777)
            dataloader.shuffle(domain='D', seed=888)

            lr = get_lr(self.init_lr, self.num_epoch, epoch, self.decay_epoch)

            G_fake_gan_loss_sum = 0.0
            cycle_F_loss_sum = 0.0
            cycle_D_loss_sum = 0.0
            D_real_gan_loss_sum = 0.0
            D_fake_gan_loss_sum = 0.0

            out_argG = [self.fake_F, self.G_D2F_optim, self.G_fake_gan_loss, self.cycle_F_loss, self.cycle_D_loss]
            out_argmG = [self.fake_F, self.G_D2F_optim, self.G_fake_gan_loss, self.cycle_F_loss, self.cycle_D_loss, self.G_sum, self.img_sum]
            out_argD = [self.D_F_optim, self.D_real_gan_loss, self.D_fake_gan_loss]
            out_argmD = [self.D_F_optim, self.D_real_gan_loss, self.D_fake_gan_loss, self.D_sum]

            for step in tqdm(range(self.num_step_train), desc='Step'):
                real_F, mask_F, real_D, mask_D = dataloader.getBatch_magnitude(step * self.batch_size, (step + 1) * self.batch_size)
                feed_dictG = {self.real_F: real_F, self.real_D: real_D, self.mask_F: mask_F, self.mask_D: mask_D, self.lr: lr}

                if step % disp_step == 0:
                    fake_F, _, G_fake_gan_loss, cycle_F_loss, cycle_D_loss, G_sum, img_sum = self.sess.run(out_argmG, feed_dictG)
                    self.writer.add_summary(G_sum, epoch * self.disp_div_N + disp_cnt)
                    self.writer.add_summary(img_sum, epoch * self.disp_div_N + disp_cnt)
                else:
                    fake_F, _, G_fake_gan_loss, cycle_F_loss, cycle_D_loss = self.sess.run(out_argG, feed_dictG)

                fake_F = self.pool(fake_F)

                feed_dictD = {self.real_F: real_F, self.fake_F_pool: fake_F, self.lr: lr}

                if step % disp_step == 0:
                    _, D_real_gan_loss, D_fake_gan_loss, D_sum = self.sess.run(out_argmD, feed_dictD)
                    self.writer.add_summary(D_sum, epoch * self.disp_div_N + disp_cnt)
                    disp_cnt += 1
                else:
                    _, D_real_gan_loss, D_fake_gan_loss = self.sess.run(out_argD, feed_dictD)

                G_fake_gan_loss_sum += G_fake_gan_loss
                cycle_F_loss_sum += cycle_F_loss
                cycle_D_loss_sum += cycle_D_loss
                D_real_gan_loss_sum += D_real_gan_loss
                D_fake_gan_loss_sum += D_fake_gan_loss

            G_fake_gan_avg_loss = G_fake_gan_loss_sum / self.num_step_train
            cycle_F_avg_loss = cycle_F_loss_sum / self.num_step_train
            cycle_D_avg_loss = cycle_D_loss_sum / self.num_step_train
            D_real_gan_avg_loss = D_real_gan_loss_sum / self.num_step_train
            D_fake_gan_avg_loss = D_fake_gan_loss_sum / self.num_step_train

            feed_dict = {self.lr: lr, self.G_fake_gan_avg_loss: G_fake_gan_avg_loss,
                         self.cycle_F_avg_loss: cycle_F_avg_loss, self.cycle_D_avg_loss: cycle_D_avg_loss,
                         self.D_real_gan_avg_loss: D_real_gan_avg_loss, self.D_fake_gan_avg_loss: D_fake_gan_avg_loss}

            epoch_sum = self.sess.run(self.epoch_sum, feed_dict)
            self.writer.add_summary(epoch_sum, epoch + 1)

            if (epoch + 1) % self.save_epoch == 0:
                self.saver.save(self.sess, join(self.ckpt_dir, 'model.ckpt'), global_step=epoch + 1)

    def test(self, dataloader):
        latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        self.saver.restore(self.sess, latest_ckpt)

        save_path = join(self.save_path, self.experiment_name, 'test_N={}'.format(self.N))

        if not isdir(save_path):
            makedirs(save_path)

        for step in tqdm(range(self.num_step_test), desc='Step'):
            real_D, scale_D = dataloader.getBatch_magnitude_test(step)

            feed_dict = {self.real_D: real_D}
            test_output = np.squeeze(self.sess.run(self.fake_F, feed_dict) * scale_D)
            test_output = np.mean(test_output, axis=0)

            subpath = dataloader.flist_D[step].split('test/')[1]
            subname, fname = subpath.split('/')

            test_output = {'data': test_output}

            sub_save_path = join(save_path, subname)
            if not isdir(sub_save_path):
                makedirs(sub_save_path)

            sio.savemat(join(sub_save_path, fname), test_output)


class MR_motion_complex:
    def __init__(self, sess, opt):
        """
        Model for MR motion reduction using bootstrap subsampling and aggregation (for complex image)
        :param sess: tf session
        :param opt: options
        """
        GPU = '/device:GPU:' + str(opt.gpu_ids[0])
        print(GPU)

        self.GPU = GPU
        self.sess = sess
        self.disp_div_N = opt.disp_div_N
        self.batch_size = opt.batch_size
        self.nY = opt.nY
        self.nX = opt.nX
        self.nC = opt.nC
        self.lambda_cycle = opt.lambda_cycle
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.num_epoch = opt.num_epoch
        self.decay_epoch = opt.decay_epoch
        self.N = opt.N
        self.init_lr = opt.init_lr
        self.save_epoch = opt.save_epoch
        self.save_path = opt.save_path
        self.experiment_name = opt.experiment_name
        self.opt = opt

        self.num_step_train = opt.num_step_train
        self.num_step_test = opt.num_step_test
        self.ckpt_dir = opt.ckpt_dir

        self._build_model()
        self.saver = tf.train.Saver(name='saver')
        self.writer = tf.summary.FileWriter(opt.log_dir, self.sess.graph)
        self.pool = ImagePool(opt.pool_size)

    def _build_model(self):
        with tf.device(self.GPU):
            self.G_D2F = Unet(self.opt, name='Gen_D2F')
            self.G_F2D = Downsampling(name='Gen_F2D')
            self.D_F = PatchGAN(self.opt, name='Dis_F')

            self.real_F = tf.placeholder(tf.float32, [None, self.nY, self.nX, self.nC], name='real_F')
            self.real_D = tf.placeholder(tf.float32, [None, self.nY, self.nX, self.nC], name='real_D')
            self.mask_F = tf.placeholder(tf.float32, [None, self.nY, self.nX, int(self.nC / 2)], name='mask_F')
            self.mask_D = tf.placeholder(tf.float32, [None, self.nY, self.nX, int(self.nC / 2)], name='mask_D')

            self.fake_F = self.G_D2F(self.real_D)
            self.fake_D = self.G_F2D(self.real_F, self.mask_F)

            self.recon_F = self.G_D2F(self.fake_D)
            self.recon_D = self.G_F2D(self.fake_F, self.mask_D)

            self.dis_real_F = self.D_F(tf_ri2ssos(self.real_F))
            self.dis_fake_F = self.D_F(tf_ri2ssos(self.fake_F))

            self.G_fake_gan_loss = LSGAN_loss(self.dis_fake_F, tf.ones_like(self.dis_fake_F))

            self.cycle_F_loss = cycle_consistency_loss(tf_ri2ssos(self.real_F), tf_ri2ssos(self.recon_F))
            self.cycle_D_loss = cycle_consistency_loss(tf_ri2ssos(self.real_D), tf_ri2ssos(self.recon_D))

            self.G_total_loss = self.G_fake_gan_loss + self.lambda_cycle * (self.cycle_F_loss + self.cycle_D_loss)

            self.fake_F_pool = tf.placeholder(tf.float32, [None, self.nY, self.nX, self.nC], name='fake_F_pool')
            self.dis_fake_F_pool = self.D_F(tf_ri2ssos(self.fake_F_pool))

            self.D_real_gan_loss = LSGAN_loss(self.dis_real_F, tf.ones_like(self.dis_real_F))
            self.D_fake_gan_loss = LSGAN_loss(self.dis_fake_F_pool, tf.zeros_like(self.dis_fake_F_pool))

            self.D_total_loss = (self.D_real_gan_loss + self.D_fake_gan_loss) / 2

            self.G_fake_gan_sum = tf.summary.scalar('G/1_fake_gan_loss', self.G_fake_gan_loss)
            self.cycle_F_sum = tf.summary.scalar('G/2_cycle_F_loss', self.cycle_F_loss)
            self.cycle_D_sum = tf.summary.scalar('G/2_cycle_D_loss', self.cycle_D_loss)
            self.G_total_sum = tf.summary.scalar('G/3_total_loss', self.G_total_loss)
            self.G_sum = tf.summary.merge([self.G_fake_gan_sum, self.cycle_F_sum, self.cycle_D_sum, self.G_total_sum])

            self.D_real_gan_sum = tf.summary.scalar('D/1_real_gan_loss', self.D_real_gan_loss)
            self.D_fake_gan_sum = tf.summary.scalar('D/1_fake_gan_loss', self.D_fake_gan_loss)
            self.D_total_sum = tf.summary.scalar('D/2_total_loss', self.D_total_loss)
            self.D_sum = tf.summary.merge([self.D_real_gan_sum, self.D_fake_gan_sum, self.D_total_sum])

            self.real_F_sum = tf.summary.image('FDF/1_real_F', convert2int(tf_ri2ssos(self.real_F)), max_outputs=1)
            self.real_D_sum = tf.summary.image('DFD/1_real_D', convert2int(tf_ri2ssos(self.real_D)), max_outputs=1)
            self.fake_F_sum = tf.summary.image('DFD/1_fake_F', convert2int(tf_ri2ssos(self.fake_F)), max_outputs=1)
            self.fake_D_sum = tf.summary.image('FDF/1_fake_D', convert2int(tf_ri2ssos(self.fake_D)), max_outputs=1)
            self.recon_F_sum = tf.summary.image('FDF/3_recon_F', convert2int(tf_ri2ssos(self.recon_F)), max_outputs=1)
            self.recon_D_sum = tf.summary.image('DFD/3_recon_D', convert2int(tf_ri2ssos(self.recon_D)), max_outputs=1)
            self.img_sum = tf.summary.merge([self.real_F_sum, self.real_D_sum, self.fake_F_sum, self.fake_D_sum, self.recon_F_sum, self.recon_D_sum])

            self.lr = tf.placeholder(tf.float32, None, name='lr')
            self.G_fake_gan_avg_loss = tf.placeholder(tf.float32, None, name='G_fake_gan_avg_loss')
            self.cycle_F_avg_loss = tf.placeholder(tf.float32, None, name='cycle_F_avg_loss')
            self.cycle_D_avg_loss = tf.placeholder(tf.float32, None, name='cycle_D_avg_loss')
            self.D_real_gan_avg_loss = tf.placeholder(tf.float32, None, name='D_real_gan_avg_loss')
            self.D_fake_gan_avg_loss = tf.placeholder(tf.float32, None, name='D_fake_gan_avg_loss')

            self.lr_sum = tf.summary.scalar('Epoch/1_lr', self.lr)
            self.G_fake_gan_avg_sum = tf.summary.scalar('Epoch/2_G_fake_gan_avg_loss', self.G_fake_gan_avg_loss)
            self.cycle_F_avg_sum = tf.summary.scalar('Epoch/3_cycle_F_avg_loss', self.cycle_F_avg_loss)
            self.cycle_D_avg_sum = tf.summary.scalar('Epoch/3_cycle_D_avg_loss', self.cycle_D_avg_loss)
            self.D_real_gan_avg_sum = tf.summary.scalar('Epoch/4_D_real_gan_avg_loss', self.D_real_gan_avg_loss)
            self.D_fake_gan_avg_sum = tf.summary.scalar('Epoch/4_D_fake_gan_avg_loss', self.D_fake_gan_avg_loss)
            self.epoch_sum = tf.summary.merge([self.lr_sum, self.G_fake_gan_avg_sum, self.cycle_F_avg_sum, self.cycle_D_avg_sum, self.D_real_gan_avg_sum, self.D_fake_gan_avg_sum])

            self.G_D2F_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2).minimize(self.G_total_loss, var_list=self.G_D2F.variables)
            self.D_F_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2).minimize(self.D_total_loss, var_list=self.D_F.variables)

    def train(self, dataloader):
        disp_step = ceil(self.num_step_train / self.disp_div_N)

        latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        if latest_ckpt is None:
            print('Start with random initialization')
            self.sess.run(tf.global_variables_initializer())
            epoch_start = 0
        else:
            print('Start from saved model - ' + latest_ckpt)
            self.saver.restore(self.sess, latest_ckpt)
            epoch_start = myNumExt(latest_ckpt)

        for epoch in tqdm(range(epoch_start, self.num_epoch), desc='Epoch', total=self.num_epoch, initial=epoch_start):
            disp_cnt = 0

            dataloader.shuffle(domain='F', seed=777)
            dataloader.shuffle(domain='D', seed=888)

            lr = get_lr(self.init_lr, self.num_epoch, epoch, self.decay_epoch)

            G_fake_gan_loss_sum = 0.0
            cycle_F_loss_sum = 0.0
            cycle_D_loss_sum = 0.0
            D_real_gan_loss_sum = 0.0
            D_fake_gan_loss_sum = 0.0

            out_argG = [self.fake_F, self.G_D2F_optim, self.G_fake_gan_loss, self.cycle_F_loss, self.cycle_D_loss]
            out_argmG = [self.fake_F, self.G_D2F_optim, self.G_fake_gan_loss, self.cycle_F_loss, self.cycle_D_loss, self.G_sum, self.img_sum]
            out_argD = [self.D_F_optim, self.D_real_gan_loss, self.D_fake_gan_loss]
            out_argmD = [self.D_F_optim, self.D_real_gan_loss, self.D_fake_gan_loss, self.D_sum]

            for step in tqdm(range(self.num_step_train), desc='Step'):
                real_F, mask_F, real_D, mask_D = dataloader.getBatch_complex(step * self.batch_size, (step + 1) * self.batch_size)
                feed_dictG = {self.real_F: real_F, self.real_D: real_D, self.mask_F: mask_F, self.mask_D: mask_D, self.lr: lr}

                if step % disp_step == 0:
                    fake_F, _, G_fake_gan_loss, cycle_F_loss, cycle_D_loss, G_sum, img_sum = self.sess.run(out_argmG, feed_dictG)
                    self.writer.add_summary(G_sum, epoch * self.disp_div_N + disp_cnt)
                    self.writer.add_summary(img_sum, epoch * self.disp_div_N + disp_cnt)
                else:
                    fake_F, _, G_fake_gan_loss, cycle_F_loss, cycle_D_loss = self.sess.run(out_argG, feed_dictG)

                fake_F = self.pool(fake_F)

                feed_dictD = {self.real_F: real_F, self.fake_F_pool: fake_F, self.lr: lr}

                if step % disp_step == 0:
                    _, D_real_gan_loss, D_fake_gan_loss, D_sum = self.sess.run(out_argmD, feed_dictD)
                    self.writer.add_summary(D_sum, epoch * self.disp_div_N + disp_cnt)
                    disp_cnt += 1
                else:
                    _, D_real_gan_loss, D_fake_gan_loss = self.sess.run(out_argD, feed_dictD)

                G_fake_gan_loss_sum += G_fake_gan_loss
                cycle_F_loss_sum += cycle_F_loss
                cycle_D_loss_sum += cycle_D_loss
                D_real_gan_loss_sum += D_real_gan_loss
                D_fake_gan_loss_sum += D_fake_gan_loss

            G_fake_gan_avg_loss = G_fake_gan_loss_sum / self.num_step_train
            cycle_F_avg_loss = cycle_F_loss_sum / self.num_step_train
            cycle_D_avg_loss = cycle_D_loss_sum / self.num_step_train
            D_real_gan_avg_loss = D_real_gan_loss_sum / self.num_step_train
            D_fake_gan_avg_loss = D_fake_gan_loss_sum / self.num_step_train

            feed_dict = {self.lr: lr, self.G_fake_gan_avg_loss: G_fake_gan_avg_loss,
                         self.cycle_F_avg_loss: cycle_F_avg_loss, self.cycle_D_avg_loss: cycle_D_avg_loss,
                         self.D_real_gan_avg_loss: D_real_gan_avg_loss, self.D_fake_gan_avg_loss: D_fake_gan_avg_loss}

            epoch_sum = self.sess.run(self.epoch_sum, feed_dict)
            self.writer.add_summary(epoch_sum, epoch + 1)

            if (epoch + 1) % self.save_epoch == 0:
                self.saver.save(self.sess, join(self.ckpt_dir, 'model.ckpt'), global_step=epoch + 1)

    def test(self, dataloader):
        latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        self.saver.restore(self.sess, latest_ckpt)

        save_path = join(self.save_path, self.experiment_name, 'test_N={}'.format(self.N))

        if not isdir(save_path):
            makedirs(save_path)

        for step in tqdm(range(self.num_step_test), desc='Step'):
            real_D, scale_D = dataloader.getBatch_complex_test(step)

            feed_dict = {self.real_D: real_D}
            test_output = np.squeeze(self.sess.run(self.fake_F, feed_dict) * scale_D)
            test_output = np.mean(test_output, axis=0)

            subpath = dataloader.flist_D[step].split('test/')[1]
            subname, fname = subpath.split('/')

            test_output = {'data': test_output}

            sub_save_path = join(save_path, subname)
            if not isdir(sub_save_path):
                makedirs(sub_save_path)

            sio.savemat(join(sub_save_path, fname), test_output)
