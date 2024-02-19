import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[1, 2, 1],
                    [0, 0, 0],
                    [-1,-2,-1]]
        kernel_h = [[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = kernel_h
        self.weight_v = kernel_v

    def forward(self, x):
        x0 = x[:, 0]  # b,c,h,w
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v.to(x.device), padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h.to(x.device), padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v.to(x.device), padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h.to(x.device), padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v.to(x.device), padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h.to(x.device), padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x



