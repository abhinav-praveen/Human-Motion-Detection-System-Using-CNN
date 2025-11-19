from video_loading import VideoFrameDataset  # Make sure this is defined or imported correctly

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Define your class list
classes = ['walking', 'running', 'no_motion']

# Set up video transforms – must match your dataset's expected input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Create dataset object
dataset_path = 'dataset'  # or wherever your video folders are
dataset = VideoFrameDataset(dataset_path, classes, frames_per_video=10, transform=transform)

# Define the CNN model
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

# Set up device and initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FrameCNN(num_classes=len(classes)).to(device)

# Training setup
batch_size = 4
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for frames, labels in loader:
        # frames: (batch_size, frames_per_video, 1, 64, 64)
        batch_frames = frames.view(-1, 1, 64, 64).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_frames)  # (batch_size * frames_per_video, num_classes)

        n_frames = frames.shape[1]  # actual frames per video in this batch
        actual_batch_size = frames.shape[0]
        outputs = outputs.view(actual_batch_size, n_frames, len(classes))

        outputs = torch.mean(outputs, dim=1)  # (batch_size, num_classes)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    accuracy = correct / total
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

torch.save(model.state_dict(), 'model.pth')
