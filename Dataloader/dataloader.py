import numpy as np
import random
from os import listdir
from os.path import join
from scipy import io as sio
from Utils.utils import ri2ssos, ri2complex, complex2ri, fft2c, ifft2c


class Dataloader:
    def __init__(self, opt, phase):
        """
        dataloader
        :param opt: options
        :param phase: string, train or test
        """
        self.data_root = opt.data_root
        self.phase_root = join(self.data_root, phase)

        flist = []
        for aSub in sorted(listdir(self.phase_root)):
            sub_root = join(self.phase_root, aSub)
            for aImg in sorted(listdir(sub_root)):
                flist.append(join(sub_root, aImg))
        self.flist_F = flist
        self.flist_D = flist

        self.nY = opt.nY
        self.nX = opt.nX
        self.nC = opt.nC
        self.R = opt.R
        self.N = opt.N
        self.augmentation = opt.augmentation
        self.phase = phase
        self.len = len(self.flist_F)

    def augment_data(self, inp, mask):
        """
        Augment data
        :param inp: 3D (HWC) input numpy array
        :param mask: 3D (HWC) input numpy array
        :return: 3D (HWC) output numpy array
        """
        p_flip = np.random.rand(1)
        if p_flip > 0.85:
            inp = np.flip(inp, 1)
            mask = np.flip(mask, 1)
        elif p_flip < 0.15:
            inp = np.flip(inp, 0)
            mask = np.flip(mask, 0)
        return inp, mask

    def shuffle(self, domain, seed=0):
        """
        Shuffle the list of data
        :param domain: string, F (fully sampled) or D (downsampled)
        :param seed: int, random seed
        """
        random.seed(seed)
        if domain == 'F':
            random.shuffle(self.flist_F)
        else:
            random.shuffle(self.flist_D)

    @staticmethod
    def read_mat(filename):
        """
        Read data from .mat file
        :param filename: the name of the file
        :return: 3D (HWC) output numpy array
        """
        mat = sio.loadmat(filename, verify_compressed_data_integrity=False)
        return mat['data']

    @staticmethod
    def make_mask(inp, R):
        """
        Make subsampling mask (1D Cartesian trajectory, Gaussian random sampling)
        :param inp: 3D (HWC) output numpy array
        :param R: integer, downsampling rate
        :return: 3D (HWC) output numpy array
        """
        nY = np.shape(inp)[0]
        nX = np.shape(inp)[1]
        mask = np.zeros((nY, nX), dtype=np.float32)

        nACS = round(nY / (R ** 2))
        ACS_s = round((nY - nACS) / 2)
        ACS_e = ACS_s + nACS
        mask[ACS_s:ACS_e, :] = 1

        nSamples = int(nY / R)
        r = np.floor(np.random.normal(nY / 2, 70, nSamples))
        r = np.clip(r.astype(int), 0, nY - 1)
        mask[r.tolist(), :] = 1
        return mask

    def getBatch_magnitude(self, start, end):
        """
        Make and return batch data (for magnitude image)
        :param start: integer, the first index of the file list
        :param end: integer, the last index of the file list
        :return: 4D (BHWC) output numpy array
        """
        end = min([end, self.len])
        batch_F = self.flist_F[start:end]
        batch_D = self.flist_D[start:end]

        size_input = [end - start, self.nY, self.nX, self.nC]
        size_mask = size_input

        Input_F = np.empty(size_input, dtype=np.float32)
        Mask_F = np.empty(size_mask, dtype=np.float32)

        Input_D = np.empty(size_input, dtype=np.float32)
        Mask_D = np.empty(size_mask, dtype=np.float32)

        for iB in range(len(batch_F)):
            aInput_F = self.read_mat(batch_F[iB])  # magnitude image
            aInput_D = self.read_mat(batch_D[iB])  # magnitude image

            aMask_F = self.make_mask(aInput_F, self.R)
            aMask_D = self.make_mask(aInput_D, self.R)

            k_orig_F = np.fft.fftshift(np.fft.fft2(aInput_F))
            k_down_F = k_orig_F * aMask_F
            tmp = np.abs(np.fft.ifft2(np.fft.ifftshift(k_down_F)))

            k_orig_D = np.fft.fftshift(np.fft.fft2(aInput_D))
            k_down_D = k_orig_D * aMask_D
            aInput_D = np.abs(np.fft.ifft2(np.fft.ifftshift(k_down_D)))

            aScale_F = np.std(tmp)
            aInput_F = aInput_F / aScale_F
            aScale_D = np.std(aInput_D)
            aInput_D = aInput_D / aScale_D

            if self.phase == 'train' and self.augmentation:
                aInput_F, aMask_F = self.augment_data(aInput_F, aMask_F)
                aInput_D, aMask_D = self.augment_data(aInput_D, aMask_D)

            Input_F[iB, :, :, 0] = aInput_F
            Input_D[iB, :, :, 0] = aInput_D
            Mask_F[iB, :, :, 0] = aMask_F
            Mask_D[iB, :, :, 0] = aMask_D

        return Input_F, Mask_F, Input_D, Mask_D

    def getBatch_magnitude_test(self, idx):
        """
        Make and return batch data (for magnitude image, inference)
        :param idx: integer, the index of the file list
        :return: 4D (BHWC) output numpy array
        """
        batch_D = self.flist_D[idx]

        size_input = [self.N, self.nY, self.nX, self.nC]

        Input_D = np.empty(size_input, dtype=np.float32)
        Scale_D = np.empty(size_input, dtype=np.float32)

        for iB in range(self.N):
            aInput_D = self.read_mat(batch_D)  # magnitude image with motion artifact

            aMask_D = self.make_mask(aInput_D, self.R)

            k_orig_D = np.fft.fftshift(np.fft.fft2(aInput_D))
            k_down_D = k_orig_D * aMask_D
            aInput_D = np.abs(np.fft.ifft2(np.fft.ifftshift(k_down_D)))

            aScale_D = np.std(aInput_D)
            aInput_D = aInput_D / aScale_D

            if self.phase == 'train' and self.augmentation:
                aInput_D, aMask_D = self.augment_data(aInput_D, aMask_D)

            Input_D[iB, :, :, 0] = aInput_D
            Scale_D[iB, :, :, 0] = np.tile(aScale_D, [self.nY, self.nX])

        return Input_D, Scale_D

    def getBatch_complex(self, start, end):
        """
        Make and return batch data (for complex image)
        :param start: integer, the first index of the file list
        :param end: integer, the last index of the file list
        :return: 4D (BHWC) output numpy array
        """
        end = min([end, self.len])
        batch_F = self.flist_F[start:end]
        batch_D = self.flist_D[start:end]

        size_input = [end - start, self.nY, self.nX, self.nC]
        size_mask = [end - start, self.nY, self.nX, int(self.nC / 2)]

        Input_F = np.empty(size_input, dtype=np.float32)
        Mask_F = np.empty(size_mask, dtype=np.float32)

        Input_D = np.empty(size_input, dtype=np.float32)
        Mask_D = np.empty(size_mask, dtype=np.float32)

        for iB in range(len(batch_F)):
            aInput_F = self.read_mat(batch_F[iB])  # concatenated real/imaginary image
            aInput_D = self.read_mat(batch_D[iB])  # concatenated real/imaginary image

            aMask_F = self.make_mask(aInput_F, self.R)
            aMask_D = self.make_mask(aInput_D, self.R)

            aMask_F = np.tile(aMask_F[:, :, np.newaxis], [1, 1, int(self.nC / 2)])
            aMask_D = np.tile(aMask_D[:, :, np.newaxis], [1, 1, int(self.nC / 2)])

            k_orig_F = np.fft.fftshift(fft2c(ri2complex(aInput_F)), axes=[0, 1])
            k_down_F = k_orig_F * aMask_F
            tmp = complex2ri(ifft2c(np.fft.ifftshift(k_down_F, axes=[0, 1])))

            k_orig_D = np.fft.fftshift(fft2c(ri2complex(aInput_D)), axes=[0, 1])
            k_down_D = k_orig_D * aMask_D
            aInput_D = complex2ri(ifft2c(np.fft.ifftshift(k_down_D, axes=[0, 1])))

            aScale_F = np.std(ri2ssos(tmp))
            aInput_F = aInput_F / aScale_F
            aScale_D = np.std(ri2ssos(aInput_D))
            aInput_D = aInput_D / aScale_D

            if self.phase == 'train' and self.augmentation:
                aInput_F, aMask_F = self.augment_data(aInput_F, aMask_F)
                aInput_D, aMask_D = self.augment_data(aInput_D, aMask_D)

            Input_F[iB, :, :, :] = aInput_F
            Input_D[iB, :, :, :] = aInput_D
            Mask_F[iB, :, :, :] = aMask_F
            Mask_D[iB, :, :, :] = aMask_D

        return Input_F, Mask_F, Input_D, Mask_D

    def getBatch_complex_test(self, idx):
        """
        Make and return batch data (for complex image, inference)
        :param idx: integer, the index of the file list
        :return: 4D (BHWC) output numpy array
        """
        batch_D = self.flist_D[idx]

        size_input = [self.N, self.nY, self.nX, self.nC]

        Input_D = np.empty(size_input, dtype=np.float32)
        Scale_D = np.empty(size_input, dtype=np.float32)

        for iB in range(self.N):
            aInput_D = self.read_mat(batch_D)  # concatenated real/imaginary image with motion artifact

            aMask_D = self.make_mask(aInput_D, self.R)

            aMask_D = np.tile(aMask_D[:, :, np.newaxis], [1, 1, int(self.nC / 2)])

            k_orig_D = np.fft.fftshift(fft2c(ri2complex(aInput_D)), axes=[0, 1])
            k_down_D = k_orig_D * aMask_D
            aInput_D = complex2ri(ifft2c(np.fft.ifftshift(k_down_D, axes=[0, 1])))

            aScale_D = np.std(ri2ssos(aInput_D))
            aInput_D = aInput_D / aScale_D

            if self.phase == 'train' and self.augmentation:
                aInput_D, aMask_D = self.augment_data(aInput_D, aMask_D)

            Input_D[iB, :, :, :] = aInput_D
            Scale_D[iB, :, :, :] = np.tile(aScale_D, [self.nY, self.nX, self.nC])

        return Input_D, Scale_D
