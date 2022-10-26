
import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.view(-1, 1, 250, 250)


class Model(nn.Sequential):
    def __init__(self):
        super().__init__(
            View(),
            nn.Conv2d(stride=3, kernel_size=32, out_channels=8, in_channels=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(stride=2, kernel_size=32, out_channels=16, in_channels=8),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),

            nn.Conv2d(kernel_size=16, out_channels=32, in_channels=16, padding="same"),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),

            nn.Conv2d(kernel_size=16, out_channels=64, in_channels=32, padding="same"),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(out_features=256, in_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(out_features=1, in_features=256)
        )

        # self.conv1 = nn.Conv2d(stride=3, kernel_size=32, out_channels=8, in_channels=1)
        # self.conv2 = nn.Conv2d(stride=2, kernel_size=32, out_channels=16, in_channels=8)
        # self.conv3 = nn.Conv2d(kernel_size=16, out_channels=32, in_channels=16, padding="same")
        # self.conv4 = nn.Conv2d(kernel_size=16, out_channels=64, in_channels=32, padding="same")
        #
        # self.pool = nn.MaxPool2d(2, 2)
        # self.dropOut = nn.Dropout(0.2)
        # self.flatten = nn.Flatten()
        #
        # self.dense1 = nn.Linear(out_features=256, in_features=256)
        # self.dense2 = nn.Linear(out_features=1, in_features=256)
        #
        # self.batchNorm1 = nn.BatchNorm2d(num_features=8)
        # self.batchNorm2 = nn.BatchNorm2d(num_features=16)
        # self.batchNorm3 = nn.BatchNorm2d(num_features=32)
        # self.batchNorm4 = nn.BatchNorm2d(num_features=64)

    # def forward(self, inputs):
    #     tensor = inputs.view(-1, 1, 250, 250)
    #
    #     tensor = self.conv1(tensor)
    #     tensor = self.batchNorm1(tensor)
    #     tensor = torch.relu(tensor)
    #     tensor = self.dropOut(tensor)
    #
    #     tensor = self.conv2(tensor)
    #     tensor = self.batchNorm2(tensor)
    #     tensor = torch.relu(tensor)
    #     tensor = self.pool(tensor)
    #     tensor = self.dropOut(tensor)
    #
    #     tensor = self.conv3(tensor)
    #     tensor = self.batchNorm3(tensor)
    #     tensor = torch.relu(tensor)
    #     tensor = self.pool(tensor)
    #     tensor = self.dropOut(tensor)
    #
    #     tensor = self.conv4(tensor)
    #     tensor = self.batchNorm4(tensor)
    #     tensor = torch.relu(tensor)
    #     tensor = self.pool(tensor)
    #
    #     tensor = self.flatten(tensor)
    #     tensor = self.dense1(tensor)
    #     tensor = torch.relu(tensor)
    #     tensor = self.dense2(tensor)
    #     return tensor
