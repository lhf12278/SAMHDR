import numpy as np
import os, glob
import cv2
from math import log10
import time
import torch
import skimage.metrics as sk
import math
import torchvision.transforms as transforms
from bisect import bisect_right
import torch
import torch.nn as nn

def ReadExpoTimes(fileName):
    return np.power(2, np.loadtxt(fileName)) #2 的 __次方


def list_all_files_sorted(folderName, extension=""):
    return sorted(glob.glob(os.path.join(folderName, "*" + extension)))


def ReadImages(fileNames):
    imgs = []
    for imgStr in fileNames:
        img = cv2.imread(imgStr, cv2.IMREAD_UNCHANGED)
        img = np.float32(img)/255
        # img = np.float32(img)
        # img = img / 2 ** 16

        # img.clip(0, 1)

        imgs.append(img)
    return np.array(imgs)

def ReadLabel(fileName):
    label = cv2.imread(os.path.join(fileName, 'ref_hdr_aligned_linear.hdr'), flags=cv2.IMREAD_UNCHANGED)
    # label = cv2.imread(os.path.join(fileName, 'HDRImg.hdr'), flags=cv2.IMREAD_UNCHANGED)
    label = label[:, :, [2, 1, 0]]
    return label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
 ])

def ReadMs(fileName):
    # Ms = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)   # 读取灰度图
    Ms = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
    Ms = torch.from_numpy(Ms).unsqueeze(2).numpy()
    Ms = transform(Ms)  # tensor 归一化
    Ms = Ms.permute(1, 2, 0).numpy()
    return Ms

def LDR_to_HDR(imgs, expo, gamma):
    return (imgs ** gamma) / expo


def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)


def range_compressor_tensor(x):
    const_1 = torch.from_numpy(np.array(1.0)).cuda()
    const_5000 = torch.from_numpy(np.array(5000.0)).cuda()
    return (torch.log(const_1 + const_5000 * x)) / torch.log(const_1 + const_5000)

def reverse_tonemap(CompressedImage):
    return ((np.power(5001, CompressedImage)) - 1) / 5000

def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += sk.peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean, data_range):
    Img = img.data.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    Iclean = imclean.data.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += sk.structural_similarity(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range, multichannel=True)
    return (SSIM/Img.shape[0])

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 2.0 # input -1~1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

class SSIM():
    def __init__(self, range=1):
        self.range = range

    def __call__(self, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self._ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    def _ssim(self, img1, img2):
        C1 = (0.01 * self.range) ** 2
        C2 = (0.03 * self.range) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()

def print_log(epoch, num_epochs, one_epoch_time, train_psnr1, train_ssim1, train_PSNR_u1, train_SSIM_u1,train_psnr, train_ssim, train_PSNR_u, train_SSIM_u, train_loss, category):
    print('({0:.3f}s) Epoch [{1}/{2}], Train_PSNR1:{3:.2f},Train_SSIM1:{4:.4f},Train_PSNR_u1:{5:.2f},Train_SSIM_u1:{6:.4f},Train_PSNR:{7:.2f},Train_SSIM:{8:.4f},Train_PSNR_u:{9:.2f},Train_SSIM_u:{10:.4f},Train_Loss:{11:.4f},'
          .format(one_epoch_time, epoch, num_epochs, train_psnr1, train_ssim1, train_PSNR_u1, train_SSIM_u1, train_psnr, train_ssim, train_PSNR_u, train_SSIM_u, train_loss))

    # --- Write the training log --- #
    with open('./log/{}_log.txt'.format(category), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.3f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f},Train_SSIM:{5:.4f},Train_PSNR_u:{6:.2f},Train_SSIM_u:{7:.4f}, Train_Loss:{8:.4f},'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, train_ssim, train_PSNR_u, train_SSIM_u, train_loss), file=f)

def print_log_base(epoch, num_epochs, one_epoch_time,train_psnr, train_ssim, train_PSNR_u, train_SSIM_u, train_loss, category):
    print('({0:.3f}s) Epoch [{1}/{2}], Train_PSNR:{3:.4f},Train_SSIM:{4:.4f},Train_PSNR_u:{5:.4f},Train_SSIM_u:{6:.4f},Train_Loss:{7:.4f},'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, train_ssim, train_PSNR_u, train_SSIM_u, train_loss))

    # --- Write the training log --- #
    with open('./log/{}_log.txt'.format(category), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.3f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.4f},Train_SSIM:{5:.4f},Train_PSNR_u:{6:.4f},Train_SSIM_u:{7:.4f}, Train_Loss:{8:.4f},'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, train_ssim, train_PSNR_u, train_SSIM_u, train_loss), file=f)

def print_test_log(epoch, num_epochs, test_psnr, test_ssim, test_psnr_u, test_ssim_u, test_loss, category):
    print(' Epoch [{0}/{1}], Test_PSNR:{2:.2f}, Test_SSIM:{3:.4f},Test_PSNR_u:{4:.2f},Test_SSIM_u:{5:.4f},Test_loss:{6:.4f}'
          .format(epoch, num_epochs, test_psnr, test_ssim,  test_psnr_u, test_ssim_u, test_loss,))

# def print_test_log(epoch, num_epochs, test_psnr, test_ssim, test_psnr_u, test_ssim_u, category):
#     print(' Epoch [{0}/{1}], Test_PSNR:{2:.2f}, Test_SSIM:{3:.4f},Test_PSNR_u:{4:.2f},Test_SSIM_u:{5:.4f}'
#         .format(epoch, num_epochs, test_psnr, test_ssim, test_psnr_u, test_ssim_u, ))
    # --- Write the training log --- #
    with open('./log/{}_log.txt'.format(category), 'a') as f:
        print('Date: {0}s,  Epoch: [{1}/{2}], Test_PSNR_Avg: {3:.2f}, Test_SSIM_Avg: {4:.4f}, Test_PSNR_u:{5:.2f},Test_SSIM_u:{6:.4f},Test_loss:{7:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),epoch, num_epochs, test_psnr,test_ssim,test_psnr_u, test_ssim_u, test_loss), file=f)


class Logger:

    def __init__(self, log_file):
        '''/path/to/log_file.txt'''
        self.log_file = log_file

    def __call__(self, input):
        input = str(input)
        with open(self.log_file, 'a') as f:
            f.writelines(input+'\n')
        print(input)


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class PSNR():
    def __init__(self, range=1):
        self.range = range

    def __call__(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        return 20 * math.log10(self.range / math.sqrt(mse))


class SSIM():
    def __init__(self, range=1):
        self.range = range

    def __call__(self, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self._ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    def _ssim(self, img1, img2):
        C1 = (0.01 * self.range) ** 2
        C2 = (0.03 * self.range) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()
