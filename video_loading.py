import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, classes, frames_per_video=10, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.frames_per_video = frames_per_video
        self.transform = transform

        # Collect tuples of (video_path, label_index)
        self.video_list = []
        for label, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            for file in os.listdir(cls_dir):
                if file.lower().endswith('.avi'):
                    self.video_list.append((os.path.join(cls_dir, file), label))

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path, label = self.video_list[idx]
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Indices of frames to extract
        frame_indices = torch.linspace(0, total_frames - 1, steps=self.frames_per_video).long()
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx.item())
            ret, frame = cap.read()
            if not ret:
                # If reading fails, append zero tensor
                frame_tensor = torch.zeros((1, 64, 64))
            else:
                # Convert to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self.transform:
                    frame_tensor = self.transform(gray_frame)
                else:
                    # Default transform to tensor
                    frame_tensor = torch.from_numpy(gray_frame).unsqueeze(0).float() / 255.
            frames.append(frame_tensor)
        cap.release()

        # Stack frames: shape (frames_per_video, 1, 64, 64)
        frames = torch.stack(frames)
        return frames, label

# Usage
dataset_path = 'dataset'
classes = ['walking', 'running', 'no_motion']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = VideoFrameDataset(dataset_path, classes, frames_per_video=10, transform=transform)
