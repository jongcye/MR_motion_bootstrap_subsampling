import argparse
import os


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--N', type=int, default=15, help='the subsampling aggregation factor')
        self.parser.add_argument('--R', type=float, default=3, help='the downsampling rate')
        self.parser.add_argument('--augmentation', action='store_true', help='true if use data augmentation')
        self.parser.add_argument('--batch_size', type=int, default=1, help='the size of the batch')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='the moment for the optimizer, 0 to 1')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='the moment for the optimizer, 0 to 1')
        self.parser.add_argument('--data_root', type=str, default='../Data', help='the path of data')
        self.parser.add_argument('--decay_epoch', type=int, default=201, help='the epoch to start learning rate decay')
        self.parser.add_argument('--disp_div_N', type=int, default=100, help='display N times per epoch')
        self.parser.add_argument('--experiment_name', type=str, default='MR_motion_reduction', help='the name of the experiment')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='ID of GPUs')
        self.parser.add_argument('--init_lr', type=float, default=1e-4, help='the initial learning rate')
        self.parser.add_argument('--lambda_cycle', type=float, default=10, help='the lambda for cycle-consistency loss')
        self.parser.add_argument('--nC', type=int, default=1, help='the number of input channels, 1 for the magnitude image, 2 * the number of coils for complex image')
        self.parser.add_argument('--nX', type=int, default=320, help='the width of input image')
        self.parser.add_argument('--nY', type=int, default=320, help='the height of input image')
        self.parser.add_argument('--ndf', type=int, default=64, help='the number of discriminator filters of the first layer')
        self.parser.add_argument('--ngf', type=int, default=64, help='the number of generator filters of the first layer')
        self.parser.add_argument('--num_epoch', type=int, default=200, help='the number of epochs')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of the image pool')
        self.parser.add_argument('--save_epoch', type=int, default=10, help='save the model per N epochs')
        self.parser.add_argument('--save_path', type=str, default='../Results', help='the path for saving results')
        self.parser.add_argument('--training', action='store_true', help='true if training mode')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        print('---------- Options ----------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('------------ End ------------')

        expr_dir = os.path.join(self.opt.save_path, self.opt.experiment_name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('---------- Options ----------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('------------ End ------------')
        opt_file.close()
        return self.opt
