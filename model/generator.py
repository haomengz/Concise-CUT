import torch
import torch.nn as nn
import config
from .sample_layer import Upsample, Downsample
from .resnet_block import ResnetBlock

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64, residuals=9):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, features, kernel_size=7),
            nn.InstanceNorm2d(features),
            nn.ReLU(True)
        ]
        features_prev = features

        for i in range(2):
            features *= 2
            layers += [
                nn.Conv2d(features_prev, features, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(features),
                nn.ReLU(True),
                Downsample(features)
            ]
            features_prev = features

        for i in range(residuals):
            layers += [ResnetBlock(features_prev)]

        for i in range(2):
            features //= 2
            layers += [

                Upsample(features_prev),
                nn.Conv2d(features_prev, features, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(features),
                nn.ReLU(True)
            ]

            features_prev = features

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(features_prev, in_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, input, encode_only=False, patch_ids=None):
        if not encode_only:
            return(self.model(input))
        else:
            num_patches = 256
            return_ids = []
            return_feats = []
            feat = input
            mlp_id = 0

            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)

                if layer_id in [0, 4, 8, 12, 16]:
                    B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
                    feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)

                    if patch_ids is not None:
                        patch_id = patch_ids[mlp_id]
                        mlp_id += 1
                    else:
                        patch_id = torch.randperm(feat_reshape.shape[1], device=config.DEVICE) #, device=config.DEVICE
                        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))] # .to(patch_ids.device)
                        return_ids.append(patch_id)
                    x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])

                    return_feats.append(x_sample)
            return return_feats, return_ids
