import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import torchvision
import torchvision.models as models


class TwoLayerMLPConv(nn.Module):
    def __init__(self, in_channels=512, hidden_dim=256, out_channels=256):
        super(TwoLayerMLPConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.mlp(x)  # 直接在 (batch, channels, height, width) 上操作


class CrossAttentionWithFFN(nn.Module):
    def __init__(self, batch_size, d_model=256, ffn_hidden=512):
        super(CrossAttentionWithFFN, self).__init__()
        self.d_model = d_model
        self.batch_size = batch_size

        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 前馈神经网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model)
        )

    def cross_attention_enhance(self, TAI_feature, Deep_feature):
        # 变换形状为 [batch, seq_len, channels]
        TAI_feature = TAI_feature.permute(0, 2, 3, 1).reshape(self.batch_size, 24 * 77, self.d_model)
        Deep_feature = Deep_feature.permute(0, 2, 3, 1).reshape(self.batch_size, 24 * 77, self.d_model)

        # 计算 Q, K, V
        Q = self.W_q(TAI_feature)
        K = self.W_k(Deep_feature)
        V = self.W_v(Deep_feature)

        # 计算注意力
        d_k = self.d_model ** 0.5
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 计算加权和
        enhanced_feature = torch.matmul(attention_weights, V)

        # 前馈神经网络
        enhanced_feature = self.ffn(enhanced_feature)

        # 恢复形状为 [8, 256, 24, 77]
        enhanced_feature = enhanced_feature.reshape(self.batch_size, 24, 77, self.d_model).permute(0, 3, 1, 2)

        return enhanced_feature


class ResnetEncoder(nn.Module):
    def __init__(self, num_layers=18, pretrained=True):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        res_feat_chs = {18: 256,
                        34: 256,
                        50: 1024}

        self.res_feat_chs = res_feat_chs[num_layers]

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.location_xy = nn.Sequential(
            nn.Linear(self.res_feat_chs * 7 * 7, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 2),
        )
        self.location_z = nn.Sequential(
            nn.Linear(self.res_feat_chs * 7 * 7, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
        )
        self.orientation_conf = nn.Sequential(
            nn.Linear(self.res_feat_chs * 7 * 7, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 2),
        )

    # TAI_feature 是 pretrain encoder 输出的feature
    # Deep_feature 是 train encoder 输出的feature
    def cross_attention_enhance(self, TAI_feature, Deep_feature):
        # 变换形状为 [batch, seq_len, channels]
        batch_size = TAI_feature.shape[0]
        TAI_feature = TAI_feature.permute(0, 2, 3, 1).reshape(batch_size, 24 * 77, 256)
        Deep_feature = Deep_feature.permute(0, 2, 3, 1).reshape(batch_size, 24 * 77, 256)

        # 线性投影层
        d_model = 256
        W_q = nn.Linear(d_model, d_model)
        W_k = nn.Linear(d_model, d_model)
        W_v = nn.Linear(d_model, d_model)

        # 计算 Q, K, V
        Q = W_q(TAI_feature)
        K = W_k(Deep_feature)
        V = W_v(Deep_feature)

        # 计算注意力
        d_k = d_model ** 0.5
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 计算加权和
        enhanced_feature = torch.matmul(attention_weights, V)

        # 恢复形状为 [8, 256, 24, 77]
        enhanced_feature = enhanced_feature.reshape(batch_size, 24, 77, 256).permute(0, 3, 1, 2)

        return enhanced_feature


    def forward(self, input_image, bbox, pretrained_encoder_feature, train=True):
        self.features = []
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        last_feat = self.features[-1]  # [8,256,24,77]

        mse_loss = 0
        if train:
            # concat
            # concat_feature = torch.cat((last_feat, pretrained_encoder_feature), dim=1)
            # # conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
            # # enhanced_last_feat = conv(concat_feature)
            # two_layer_mlp = TwoLayerMLPConv()
            # enhanced_last_feat = two_layer_mlp(concat_feature)

            # MLP
            # two_layer_mlp = TwoLayerMLPConv()
            # enhanced_last_feature = two_layer_mlp(pretrained_encoder_feature)

            # cross attention enhance
            # enhanced_last_feat = self.cross_attention_enhance(pretrained_encoder_feature, last_feat)

            # cross attention + fnn enhance
            batch_size = last_feat.shape[0]
            cross_attention_fnn = CrossAttentionWithFFN(batch_size=batch_size)
            enhanced_last_feat = cross_attention_fnn.cross_attention_enhance(pretrained_encoder_feature, last_feat)

            # 计算 loss
            mse_loss_fn = nn.MSELoss()
            mse_loss = mse_loss_fn(last_feat, enhanced_last_feat)

        if len(bbox.shape) == 3:
            f = torchvision.ops.roi_align(last_feat, [i/16 for i in bbox], (7, 7)) # [24,256,7,7]
            # roi_pretrained_encoder_feature = torchvision.ops.roi_align(pretrained_encoder_feature, [i/16 for i in bbox], (7, 7))
        else:
            f = torchvision.ops.roi_align(last_feat, [bbox/16], (7, 7)) # [24,256,7,7]
            # roi_pretrained_encoder_feature = torchvision.ops.roi_align(pretrained_encoder_feature, [bbox/16], (7, 7))

        # mse_loss_fn = nn.MSELoss()
        # mse_loss = mse_loss_fn(f, roi_pretrained_encoder_feature)

        f = f.view(-1, self.res_feat_chs * 7 * 7) # [24,256x7x7]

        location_xy = self.location_xy(f)
        location_xy = location_xy.view(-1, 2)

        location_z = self.location_z(f)
        orientation_conf = self.orientation_conf(f)

        return location_xy, location_z, orientation_conf, mse_loss
