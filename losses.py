import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import utils


def align_loss(x, y, alpha=2):
    '''
    https://github.com/SsnL/align_uniform/blob/master/align_uniform/__init__.py
    :param x:
    :param y:
    :param alpha:
    :return:
    '''
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


class InstanceAlignment(nn.Module):
    def __init__(self, num_crops):
        super(InstanceAlignment, self).__init__()
        self.num_crops = num_crops

    def forward(self, x):
        x_list = x.chunk(self.num_crops)
        total_loss = 0
        n_loss_term = 0
        for ix, x in enumerate(x_list):
            for i in range(ix+1, len(x_list)):
                total_loss += align_loss(x, x_list[i])
                n_loss_term += 1
        total_loss /= n_loss_term
        return total_loss


class CenterAlignment(nn.Module):
    def __init__(self, num_crops, num_classes=104, fea_dim=256, momentum=0.9):
        super(CenterAlignment, self).__init__()
        self.num_crops = num_crops
        self.num_classes = num_classes
        self.fea_dim = fea_dim
        self.momentum = momentum
        self.register_buffer("center_skt", torch.zeros(num_classes, fea_dim))
        self.register_buffer("center_img", torch.zeros(num_classes, fea_dim))

    def forward(self, x, l, modality='img'):
        class_in_batch = self.update_center(x, l, modality)
        return align_loss(self.center_img[class_in_batch], self.center_skt[class_in_batch]) * utils.get_world_size()

    def update_center(self, x, l, modality):
        self.center_img = self.center_img.detach()
        self.center_skt = self.center_skt.detach()

        with torch.no_grad():  # actually dist.all_gather has no grad, we use torch.no_grad to explictly indicate this
            all_x = [torch.zeros_like(x) for _ in range(utils.get_world_size())]
            all_l = [torch.zeros_like(l) for _ in range(utils.get_world_size())]
            dist.all_gather(all_x, x)
            dist.all_gather(all_l, l)

        all_x[utils.get_rank()] = x
        x = torch.cat(all_x)
        all_l = [lab for lab in all_l for i in range(self.num_crops)]
        all_l = torch.cat(all_l)

        classes_in_batch, sam2cls_idx, cl_sam_counts = torch.unique(all_l, return_counts=True, sorted=True, return_inverse=True)
        center_tmp = torch.zeros(len(classes_in_batch), self.fea_dim).cuda()
        for i, idx in enumerate(sam2cls_idx):
            center_tmp[idx] += x[i]
        center_tmp = center_tmp / cl_sam_counts.unsqueeze(1)

        if modality == 'img':
            self.center_img[classes_in_batch] = self.center_img[classes_in_batch] * self.momentum + center_tmp * (1 - self.momentum)
            self.center_img[classes_in_batch] /= self.center_img[classes_in_batch].norm(p=2, dim=1, keepdim=True)
        else:
            self.center_skt[classes_in_batch] = self.center_skt[classes_in_batch] * self.momentum + center_tmp * (1 - self.momentum)
            self.center_skt[classes_in_batch] /= self.center_skt[classes_in_batch].norm(p=2, dim=1, keepdim=True)

        return classes_in_batch


def uniform_loss(x, t=2):
    '''
    https://github.com/SsnL/align_uniform/blob/master/align_uniform/__init__.py
    :param x:
    :param t:
    :return:
    '''
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def LabelSmoothLoss(input, target, smoothing=0.1):
    '''
    :param input: logits
    :param target: shape like a list
    :param smoothing:
    :return:
    '''
    log_prob = F.log_softmax(input, dim=-1)
    weight = input.new_ones(input.size()) * smoothing / (input.size(-1) - 1.)
    weight.scatter_(-1, target.unsqueeze(-1), (1. - smoothing))
    loss = (-weight * log_prob).sum(dim=-1).mean()
    return loss


class LabelSmoothLossDINO(nn.Module):
    def __init__(self, num_crops, smoothing=0.0):
        super(LabelSmoothLossDINO, self).__init__()
        self.num_crops = num_crops
        self.smoothing = smoothing

    def forward(self, input, target):
        input = input.chunk(self.num_crops)
        total_loss = 0
        n_loss_terms = 0
        for i in range(len(input)):
            loss = LabelSmoothLoss(input[i], target, self.smoothing)
            total_loss += loss.mean()
            n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss
