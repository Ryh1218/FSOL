import math

import torch
import torch.nn.functional as F
from torch import nn

from models.utils import (
    build_backbone,
    build_regressor,
    crop_roi_feat,
    get_activation,
)
from utils.init_helper import initialize_from_cfg


class FSOL(nn.Module):
    def __init__(
        self,
        backbone,
        pool,
        embed_dim,
        dropout,
        activation,
        initializer=None,
    ):
        super().__init__()
        self.backbone = build_backbone(**backbone)
        self.in_conv = nn.Conv2d(
            self.backbone.out_dim, embed_dim, kernel_size=1, stride=1
        )
        self.fsmodel = FSModelBlock(
            pool=pool,
            out_stride=backbone.out_stride,
            activation=activation,
            dropout=dropout,
            embed_dim=embed_dim,
        )
        self.count_regressor = build_regressor(in_dim=embed_dim, activation=activation)
        for module in [self.in_conv, self.fsmodel, self.count_regressor]:
            initialize_from_cfg(module, initializer)

    def forward(self, input):
        image = input["image"]  # [1, 3, 512, 512]
        boxes = input["boxes"].squeeze(0)
        feat = self.in_conv(self.backbone(image))  # [1, 256, 128, 128]

        output = self.fsmodel(query_ori=feat, keys=boxes)

        density_pred = self.count_regressor(output)  # [1, 1, 512, 512]

        input.update({"density_pred": density_pred})
        return input


class FSModelBlock(nn.Module):
    def __init__(self, pool, out_stride, activation, dropout, embed_dim):
        super().__init__()
        self.pool = pool
        self.out_stride = out_stride
        self.activation = get_activation(activation)()
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.in_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.out_conv = nn.Conv2d(
            in_channels=1, out_channels=256, kernel_size=1, stride=1
        )
        self.cdc = Conv2d_Hori_Veri_Cross(
            in_channels=256, out_channels=256, kernel_size=1
        )
        self.dc = DeformConv2d(
            inc=256, outc=256, kernel_size=1, bias=None, modulation=True
        )
        self.ssp_conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.ssp_end = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, query_ori, keys):
        keys = crop_roi_feat(query_ori, keys, self.out_stride)

        attns_lst = []

        for i in keys:
            key = i

            h_p, w_p = self.pool.size
            pad = (w_p // 2, w_p // 2, h_p // 2, h_p // 2)
            _, _, h_q, w_q = query_ori.size()

            query_dc = self.activation(self.dc(query_ori))
            query_cdc = self.activation(self.cdc(query_ori))

            key_ori = F.adaptive_max_pool2d(key, self.pool.size, return_indices=False)
            key_dc = self.activation(self.dc(key_ori))
            key_cdc = self.activation(self.cdc(key_ori))

            query_lst = [
                self.process_sequence(query, h_q, w_q) for query in [query_cdc, query_dc]
            ]
            key_lst = [self.process_sequence(key, h_p, w_p) for key in [key_cdc, key_dc]]

            query = torch.stack(query_lst, dim=1).squeeze(2)
            key = torch.stack(key_lst, dim=1).squeeze(2)

            attn = self.activation(F.conv3d(F.pad(query, pad), key)).squeeze(0)

            attns_lst.append(attn)

        attns = torch.stack(attns_lst, dim=1).squeeze(0)
        attns = self.activation(self.out_conv(attn))
        attns = self.sq(query_ori, attns)
        return attns

    def process_sequence(self, sequence, h, w):
        sequence = sequence.permute(0, 2, 3, 1).contiguous()
        sequence = self.norm(sequence).permute(0, 3, 1, 2)
        return sequence.unsqueeze(0)

    def sq(self, query, similary_map):  # 128
        query = self.ssp_conv(query)
        similary_map = self.ssp_conv(similary_map)
        weight = F.cosine_similarity(query, similary_map, dim=1)  # [1, 128, 128]
        result = weight + similary_map
        result = self.ssp_end(result)
        return result


class Conv2d_Hori_Veri_Cross(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        theta=0.7,
    ):
        super(Conv2d_Hori_Veri_Cross, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 5),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.theta = theta

    def forward(self, x):
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat(
            (
                tensor_zeros,
                self.conv.weight[:, :, :, 0],
                tensor_zeros,
                self.conv.weight[:, :, :, 1],
                self.conv.weight[:, :, :, 2],
                self.conv.weight[:, :, :, 3],
                tensor_zeros,
                self.conv.weight[:, :, :, 4],
                tensor_zeros,
            ),
            2,
        )
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(
            input=x,
            weight=conv_weight,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
        )

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(
                input=x,
                weight=kernel_diff,
                bias=self.conv.bias,
                stride=self.conv.stride,
                padding=0,
                groups=self.conv.groups,
            )

            return out_normal - self.theta * out_diff


class DeformConv2d(nn.Module):
    def __init__(
        self,
        inc,
        outc,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=None,
        groups=1,
        modulation=False,
    ):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(
            inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias
        )

        self.p_conv = nn.Conv2d(
            inc,
            2 * kernel_size * kernel_size,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=groups,
        )
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(
                inc,
                kernel_size * kernel_size,
                kernel_size=3,
                padding=1,
                stride=stride,
                groups=groups,
            )
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat(
            [
                torch.clamp(p[..., :N], 0, x.size(2) - 1),
                torch.clamp(p[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        )

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (
            1 + (q_lt[..., N:].type_as(p) - p[..., N:])
        )
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
            1 - (q_rb[..., N:].type_as(p) - p[..., N:])
        )
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
            1 - (q_lb[..., N:].type_as(p) - p[..., N:])
        )
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
            1 + (q_rt[..., N:].type_as(p) - p[..., N:])
        )

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = (
            g_lt.unsqueeze(dim=1) * x_q_lt
            + g_rb.unsqueeze(dim=1) * x_q_rb
            + g_lb.unsqueeze(dim=1) * x_q_lb
            + g_rt.unsqueeze(dim=1) * x_q_rt
        )

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
        )
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = (
            index.contiguous()
            .unsqueeze(dim=1)
            .expand(-1, c, -1, -1, -1)
            .contiguous()
            .view(b, c, -1)
        )

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat(
            [
                x_offset[..., s : s + ks].contiguous().view(b, c, h, w * ks)
                for s in range(0, N, ks)
            ],
            dim=-1,
        )
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


def build_network(**kwargs):
    return FSOL(**kwargs)
