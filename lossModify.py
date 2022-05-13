import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np
from torch.nn.functional import binary_cross_entropy


class Flow_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_flows, gt_flows):
        return torch.mean(torch.abs(gen_flows - gt_flows))


class Intensity_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_frames, gt_frames):
        return torch.mean(torch.abs((gen_frames - gt_frames) ** 2))


class Gradient_Loss(nn.Module):
    def __init__(self, channels):
        super().__init__()

        pos = torch.from_numpy(np.identity(channels, dtype=np.float32))
        neg = -1 * pos
        # Note: when doing conv2d, the channel order is different from tensorflow, so do permutation.
        self.filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1).cuda()
        self.filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1).cuda()

    def forward(self, gen_frames, gt_frames):
        # Do padding to match the  result of the original tensorflow implementation
        gen_frames_x = nn.functional.pad(gen_frames, [0, 1, 0, 0])
        gen_frames_y = nn.functional.pad(gen_frames, [0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(gt_frames, [0, 1, 0, 0])
        gt_frames_y = nn.functional.pad(gt_frames, [0, 0, 0, 1])

        gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, self.filter_x))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, self.filter_y))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames_x, self.filter_x))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames_y, self.filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x + grad_diff_y)


class Adversarial_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake_outputs, alpha = 0.5):
        # TODO: compare with torch.nn.MSELoss ?
        return torch.mean(torch.log(torch.abs(fake_outputs - alpha)))


class Discriminate_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_outputs, fake_outputs, gamma = 0.1):
        return torch.mean(-1*torch.log(real_outputs) - gamma*torch.log(1 - fake_outputs))

class Dispersion_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, G_out, power=2):    
        center = torch.mean(G_out, axis=0, keepdims=True)
        distance_xy = torch.pow(torch.abs(torch.sub(G_out,center)),power)
        distance = torch.sum(distance_xy, 1)
        avg_distance = torch.mean(torch.pow(distance, 1/power))
        loss_d =torch.reciprocal(avg_distance)
        return loss_d
    
    