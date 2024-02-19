import torch
import os
import math
from time import strftime, localtime
from dataset.dataset import *
from model.model import Model
import numpy as np
import cv2
import sys
import time
device = torch.device("cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scene_directory = r'F:\yzm\data\synhdr_testing'
#scene_directory = './data/Test'
# test_file = 'model_k'
test_file = 'model_h'

psnr = PSNR()
ssim = SSIM()
def inference():
    list = os.listdir(scene_directory)
    psnr_u_sum = 0
    psnr_L_sum = 0
    ssim_L_sum = 0
    ssim_u_sum = 0
    for scene in range(len(list)):
        # expoTimes = ReadExpoTimes(os.path.join(scene_directory, list[scene], 'exposure.txt'))
        expoTimes = ReadExpoTimes(os.path.join(scene_directory, list[scene], 'input_exp.txt'))
        imgs_path = list_all_files_sorted(os.path.join(scene_directory, list[scene]), '.tif')
        imgs = ReadImages(imgs_path)
        # Ms_path = os.path.join(scene_directory, list[scene], imgs_path[1][-12:-4] + '.jpg')
        Ms_path = os.path.join(scene_directory, list[scene], imgs_path[1][-19:-4] + '.jpg')
        Ms = ReadMs(Ms_path)
        label = ReadLabel(os.path.join(scene_directory, list[scene]))  # 1000 1500 3
        label_u = range_compressor(label)  # 1000 1500 3

        pre_img0 = LDR_to_HDR(imgs[0], expoTimes[0], 2.2)
        pre_img1 = LDR_to_HDR(imgs[1], expoTimes[1], 2.2)
        pre_img2 = LDR_to_HDR(imgs[2], expoTimes[2], 2.2)
        output0 = np.concatenate((imgs[0], pre_img0), 2)
        output1 = np.concatenate((imgs[1], pre_img1), 2)
        output2 = np.concatenate((imgs[2], pre_img2), 2)

        im1 = torch.Tensor(output0).to(device)
        im1 = torch.unsqueeze(im1, 0).permute(0, 3, 1, 2)  # Change the dimensions of the tensor
        im2 = torch.Tensor(output1).to(device)
        im2 = torch.unsqueeze(im2, 0).permute(0, 3, 1, 2)
        im3 = torch.Tensor(output2).to(device)
        im3 = torch.unsqueeze(im3, 0).permute(0, 3, 1, 2)
        Ms = torch.Tensor(Ms).to(device)
        Ms = torch.unsqueeze(Ms, 0).permute(0, 3, 1, 2)

        # Load the pre-trained model
        model = Model().to(device)
        model.eval()
        model.load_state_dict(torch.load('./trained-model/{}.pkl'.format(test_file)))

        # Run
        with torch.no_grad():
            _, pre = model(im1, im2, im3, Ms)
        pre = torch.clamp(pre, 0., 1.)
        pre = pre.permute(0, 2, 3, 1)
        pre = pre.data[0].cpu().numpy()
        output = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
        cv2.imwrite('../test_result/{}.hdr'.format(scene), output)

        p_L = psnr(pre, label)
        s_L = ssim(pre, label)
        psnr_L_sum += p_L
        ssim_L_sum += s_L
        pre = range_compressor(pre)
        p_u = psnr(pre, label_u)
        s_u = ssim(pre, label_u)
        psnr_u_sum += p_u
        ssim_u_sum += s_u

    psnr_avg_u = psnr_u_sum / len(list)
    psnr_avg_L = psnr_L_sum / len(list)
    ssim_avg_L = ssim_L_sum / len(list)
    ssim_avg_u = ssim_u_sum / len(list)

    print('PSNR-L:', psnr_avg_L, 'PSNR-U:', psnr_avg_u,
          'SSIM-L:', ssim_avg_L, 'SSIM-U', ssim_avg_u)

if __name__ == '__main__':
    inference()