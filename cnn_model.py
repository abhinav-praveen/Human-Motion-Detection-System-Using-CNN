import torch.nn as nn
import torch

# Define class labels
classes = ['walking', 'running', 'no_motion']

class FrameCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(FrameCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Model initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FrameCNN(num_classes=len(classes)).to(device)
