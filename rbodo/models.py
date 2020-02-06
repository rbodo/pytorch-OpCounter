import numpy as np
import torch
import torch.nn as nn


class MlpIn(nn.Module):

    def __init__(self, input_shape, num_embed):
        super(MlpIn, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)), 4 * num_embed),
            nn.Linear(4 * num_embed, 2 * num_embed),
            nn.Linear(2 * num_embed, num_embed),
        )

    def forward(self, x):
        return self.features(x)


class MlpOut(nn.Module):

    def __init__(self, num_embed, num_frames, num_classes):
        super(MlpOut, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(num_embed * num_frames, num_embed),
            nn.Linear(num_embed, num_classes),
        )

    def forward(self, x):
        return self.features(x)


class CnnIn(nn.Module):

    def __init__(self, num_embed, num_frames, num_channels):
        super(CnnIn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, num_channels, 11, stride=(2, 1)),
            nn.Conv2d(num_channels, num_channels, 7, stride=(2, 1)),
            nn.Conv2d(num_channels, num_channels, 5, stride=(2, 1)),
            nn.Conv2d(num_channels, num_channels // 6, 3, stride=(2, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(num_frames * 3 * 42 * num_channels // 6, 4 * num_embed),
            nn.Linear(4 * num_embed, num_embed)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x)
        x = self.classifier(x)
        return x


class CnnOut(nn.Module):

    def __init__(self, num_embed, num_classes):
        super(CnnOut, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(num_embed, num_embed),
            nn.Linear(num_embed, num_classes)
        )

    def forward(self, x):
        return self.features(x)


class RnnOut(nn.Module):

    def __init__(self, num_embed, num_classes, alpha):
        super(RnnOut, self).__init__()
        self.lstm = nn.LSTM(3 * 42, alpha * num_embed, 2)
        self.classifier = nn.Sequential(
            nn.Linear(alpha * num_embed, num_embed),
            nn.Linear(num_embed, num_classes)
        )

    def forward(self, x):
        x = self.lstm(x)
        x = self.classifier(x[0])
        return x
