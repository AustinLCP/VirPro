import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import torchvision
import torchvision.models as models

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
                        50: 1024,
                        101:1024}

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

        self.text_projection = projection(512,512)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def roi_contrastive_matching(self, image_features, text_features):

        assert torch.isfinite(image_features).all(), "image_features pre-pool has NaN/Inf"

        B, n_roi, emb_dim = text_features.shape

        # feature preparation
        image_features = image_features.reshape(B*n_roi, -1)
        image_features = torch.nn.functional.adaptive_max_pool1d(image_features.unsqueeze(1), emb_dim).squeeze(1) # [24, 512]
        assert torch.isfinite(image_features).all(), "pooled image_features has NaN/Inf"
        text_features = text_features.reshape(-1, emb_dim) # [24, 512]

        # image/text feature distribution align
        text_features = self.text_projection(text_features)

        # normalized features
        image_features = image_features / (image_features.norm(dim=1, keepdim=True).clamp_min(1e-6))
        text_features = text_features / (text_features.norm(dim=1, keepdim=True).clamp_min(1e-6))

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        assert torch.isfinite(logit_scale), "logit_scale NaN/Inf"
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # Create labels (assume ground truth is a diagonal alignment)
        label_size = logits_per_text.size(0)  # a label represents an object
        labels = torch.arange(label_size).to('cuda')

        # Compute cross-entropy loss for both directions (image-to-text and text-to-image)
        loss_image_to_text = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss_text_to_image = torch.nn.functional.cross_entropy(logits_per_text, labels)

        # Average the two losses
        loss = (loss_image_to_text + loss_text_to_image) / 2

        return loss

    def forward(self, input_image, bbox):

        self.features = []
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))

        last_feat = self.features[-1]
        if len(bbox.shape) == 3:
            # for train [B,n_roi,dim]
            f = torchvision.ops.roi_align(last_feat, [i/16 for i in bbox], (7, 7))
        else:
            # for eval [n_roi,dim]
            f = torchvision.ops.roi_align(last_feat, [bbox/16], (7, 7))

        return last_feat, f


class projection(nn.Module):
    def __init__(self, in_features, out_features):
        super(projection, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x