import os
import cv2
import numpy as np

# Settings
classes = ['walking', 'running', 'no_motion']
videos_per_class = 500
video_length = 20  # number of frames per video
frame_size = (64, 64)
fps = 10  # frames per second

base_dir = 'generated_video_dataset'

# Create dataset directories
os.makedirs(base_dir, exist_ok=True)
for cls in classes:
    os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

def create_frame_diagonal_motion(size, frame_index):
    img = np.zeros(size, dtype=np.uint8)
    start = (frame_index * 2) % size[0]
    for i in range(start, size[0], 8):
        x, y = i, i
        if x < size[0] and y < size[1]:
            img[x, y] = 255
    noise = np.random.randint(0, 30, size, dtype=np.uint8)
    img = np.clip(img + noise, 0, 255)
    return img

def create_frame_vertical_motion(size, frame_index):
    img = np.zeros(size, dtype=np.uint8)
    start = (frame_index * 3) % size[0]
    for i in range(start, size[0], 6):
        img[i:i + 2, size[1]//2:size[1]//2 + 1] = 255
    noise = np.random.randint(0, 30, size, dtype=np.uint8)
    img = np.clip(img + noise, 0, 255)
    return img

def create_frame_no_motion(size):
    img = np.random.randint(0, 10, size, dtype=np.uint8)
    return img

print("Generating synthetic video dataset...")

for cls in classes:
    for v in range(videos_per_class):
        video_path = os.path.join(base_dir, cls, f'{cls}_{v}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, fps, frame_size, isColor=False)

        for f in range(video_length):
            if cls == 'walking':
                frame = create_frame_diagonal_motion(frame_size, f)
            elif cls == 'running':
                frame = create_frame_vertical_motion(frame_size, f)
            else:
                frame = create_frame_no_motion(frame_size)
            out.write(frame)

        out.release()

print("Video dataset generated in folder:", base_dir)