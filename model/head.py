import torch
import torch.nn as nn
import config
from .sample_layer import Upsample, Downsample
from .resnet_block import ResnetBlock

class Head(nn.Module):
    def __init__(self, in_channels=3, features=64, residuals=9):
        super().__init__()

        mlp = nn.Sequential(*[
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ])

        mlp_id = 0
        setattr(self, 'mlp_%d' % mlp_id, mlp)

        mlp = nn.Sequential(*[
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ])

        mlp_id = 1
        setattr(self, 'mlp_%d' % mlp_id, mlp)

        for mlp_id in range(2, 5):
            mlp = nn.Sequential(*[
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            ])
            setattr(self, 'mlp_%d' % mlp_id, mlp)

    def forward(self, feats):
        return_feats = []
        for feat_id, feat in enumerate(feats):
            mlp = getattr(self, 'mlp_%d' % feat_id)
            feat = mlp(feat)
            norm = feat.pow(2).sum(1, keepdim=True).pow(1. / 2)
            feat = feat.div(norm + 1e-7)
            return_feats.append(feat)
        return return_feats

