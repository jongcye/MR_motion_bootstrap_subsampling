import tensorflow as tf
from math import ceil
from os.path import join
from Dataloader.dataloader import Dataloader
from Model.model import MR_motion_magnitude, MR_motion_complex
from Options.options import Options

opt = Options().parse()
save_path = join(opt.save_path, opt.experiment_name)
opt.log_dir = join(save_path, 'log_dir')
opt.ckpt_dir = join(save_path, 'ckpt_dir')

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

dataloader_train = Dataloader(opt, 'train')
dataloader_test = Dataloader(opt, 'test')

opt.num_step_train = ceil(dataloader_train.len / opt.batch_size)
opt.num_step_test = dataloader_test.len

if opt.nC == 1:
    model = MR_motion_magnitude(sess, opt)
else:
    model = MR_motion_complex(sess, opt)

if opt.training:
    model.train(dataloader_train)

model.test(dataloader_test)
