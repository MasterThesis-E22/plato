
import torch
import torch.nn as nn

class EmbryosView(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input):
        return input.view(-1, 1, 250, 250)


class UpdatedEmbryosLowGPUCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            EmbryosView(),
            nn.Conv2d(stride=3, kernel_size=5, out_channels=8, in_channels=1, padding="valid"),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(stride=2, kernel_size=5, out_channels=16, in_channels=8, padding="valid"),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(kernel_size=5, out_channels=32, in_channels=16, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernel_size=5, out_channels=32, in_channels=32, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.BatchNorm2d(num_features=32),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(kernel_size=5, out_channels=64, in_channels=32, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernel_size=5, out_channels=64, in_channels=64, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.BatchNorm2d(num_features=64),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(kernel_size=5, out_channels=128, in_channels=64, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernel_size=3, out_channels=128, in_channels=128, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.BatchNorm2d(num_features=128),
        )

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(kernel_size=3, out_channels=256, in_channels=128, padding="same"),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_features=128, in_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(out_features=64, in_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(out_features=1, in_features=64)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.decoder(x)
        return x