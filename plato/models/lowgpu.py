
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
            nn.Conv2d(stride=3, kernel_size=32, out_channels=8, in_channels=1, padding="valid"),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(stride=2, kernel_size=32, out_channels=16, in_channels=8, padding="valid"),
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
        torch.nn.init.xavier_uniform_(self[1].weight)
        torch.nn.init.zeros_(self[1].bias)

        torch.nn.init.xavier_uniform_(self[5].weight)
        torch.nn.init.zeros_(self[5].bias)

        torch.nn.init.xavier_uniform_(self[10].weight)
        torch.nn.init.zeros_(self[10].bias)

        torch.nn.init.xavier_uniform_(self[15].weight)
        torch.nn.init.zeros_(self[15].bias)

        torch.nn.init.xavier_uniform_(self[20].weight)
        torch.nn.init.zeros_(self[20].bias)

        torch.nn.init.xavier_uniform_(self[23].weight)
        torch.nn.init.zeros_(self[23].bias)
