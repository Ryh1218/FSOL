import copy
import os

import torch
import torch.nn.functional as F
import torchvision
from torch import nn


class ResNet(nn.Module):
    def __init__(self, type, out_stride, out_layers, pretrained_model=""):
        super().__init__()
        self.out_stride = out_stride
        self.out_layers = out_layers
        if os.path.exists(pretrained_model):
            if type == "resnet18":
                base_dim = 64
                self.resnet = torchvision.models.resnet18(pretrained=False)
            elif type == "resnet50":
                base_dim = 256
                self.resnet = torchvision.models.resnet50(pretrained=False)
            else:
                raise NotImplementedError
            self.resnet.load_state_dict(torch.load(pretrained_model))
        else:
            if type == "resnet18":
                base_dim = 64
                self.resnet = torchvision.models.resnet18(pretrained=True)
            elif type == "resnet50":
                base_dim = 256
                self.resnet = torchvision.models.resnet50(pretrained=True)
            else:
                raise NotImplementedError
        children = list(self.resnet.children())
        self.layer0 = nn.Sequential(*children[:4])  # layer0: conv + bn + relu + pool
        self.layer1 = children[4]
        self.layer2 = children[5]
        self.layer3 = children[6]
        self.layer4 = children[7]
        planes = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        self.out_dim = sum([planes[i - 1] for i in self.out_layers])

    def forward(self, x):
        x = self.layer0(x)  # out_stride: 4
        feat1 = self.layer1(x)  # out_stride: 4
        feat2 = self.layer2(feat1)  # out_stride: 8
        feat3 = self.layer3(feat2)  # out_stride: 16
        feat4 = self.layer4(feat3)  # out_stride: 32
        feats = [feat1, feat2, feat3, feat4]
        out_strides = [4, 8, 16, 32]
        feat_list = []
        for i in self.out_layers:
            scale_factor = out_strides[i - 1] / self.out_stride # 4
            feat = feats[i - 1]
            feat = F.interpolate(feat, scale_factor=scale_factor, mode="bilinear")
            feat_list.append(feat)
        feat = torch.cat(feat_list, dim=1)
        return feat


class Regressor(nn.Module):
    def __init__(self, in_dim, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim // 2, 5, padding=2)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = nn.Conv2d(in_dim // 2, in_dim // 4, 3, padding=1)
        self.conv3 = nn.Conv2d(in_dim // 4, in_dim // 8, 1)
        self.conv4 = nn.Conv2d(in_dim // 8, 1, 1)
        self.lr = nn.LeakyReLU()
        self.r = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lr(x)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.lr(x)
        x = self.up1(x)
        x = self.conv3(x)
        x = self.lr(x)
        x = self.conv4(x)
        x = self.r(x)
        return x
        # return self.regressor(x)  # [1,1,h,w]


def crop_roi_feat(feat, boxes, out_stride):
    """
    feat: 1 x c x h x w
    boxes: m x 4, 4: [y_tl, x_tl, y_br, x_br]
    """
    _, _, h, w = feat.shape
    boxes_scaled = boxes / out_stride
    boxes_scaled[:, :2] = torch.floor(boxes_scaled[:, :2])  # y_tl, x_tl: floor
    boxes_scaled[:, 2:] = torch.ceil(boxes_scaled[:, 2:])  # y_br, x_br: ceil
    boxes_scaled[:, :2] = torch.clamp_min(boxes_scaled[:, :2], 0)
    boxes_scaled[:, 2] = torch.clamp_max(boxes_scaled[:, 2], h)
    boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], w)
    feat_boxes = []
    for idx_box in range(0, boxes.shape[0]):
        y_tl, x_tl, y_br, x_br = boxes_scaled[idx_box]
        y_tl, x_tl, y_br, x_br = int(y_tl), int(x_tl), int(y_br), int(x_br)
        feat_box = feat[:, :, y_tl : (y_br + 1), x_tl : (x_br + 1)]
        feat_boxes.append(feat_box)
    return feat_boxes


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU
    if activation == "leaky_relu":
        return nn.LeakyReLU
    raise NotImplementedError


def build_backbone(**kwargs):
    return ResNet(**kwargs)


def build_regressor(**kwargs):
    return Regressor(**kwargs)
