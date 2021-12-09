import torch
import torch.nn as nn

class Upsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.ConvTranspose2d(features, features, kernel_size=4, stride=2, padding=3)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


class Downsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3, stride=2)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)