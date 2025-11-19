import cv2
import numpy as np

# Settings to match your dataset
video_length = 20  # frames per video
frame_size = (64, 64)
fps = 10  # frames per second

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('sample_walking.avi', fourcc, fps, frame_size, isColor=False)

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

# Generate and write frames
for f in range(video_length):
    frame = create_frame_diagonal_motion(frame_size, f)
    out.write(frame)

out.release()
print("Sample walking video 'sample_walking.avi' generated.")
