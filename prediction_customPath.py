import cv2
import torch
from torchvision import transforms
from cnn_model import FrameCNN  # import your model class

def preprocess_video(video_path, frame_size=(64, 64), max_frames=10):
    """
    Reads a video file, converts frames to grayscale, resizes them,
    and extracts evenly spaced frames for prediction.
    Returns a tensor of shape (max_frames, 1, height, width).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = torch.linspace(0, total_frames - 1, steps=max_frames).long()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(frame_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    frame_buffer = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx.item())
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        input_frame = transform(gray).unsqueeze(0)  # Shape (1, 1, H, W)
        frame_buffer.append(input_frame)
    cap.release()
    if len(frame_buffer) == 0:
        raise ValueError("No frames could be read from the video.")
    return torch.cat(frame_buffer, dim=0)  # Shape (max_frames, 1, H, W)

def predict_motion(model, device, classes, video_path):
    model.eval()

    # Preprocess the video and get frame tensor
    batch_frames = preprocess_video(video_path).to(device)  # (max_frames, 1, 64, 64)

    with torch.no_grad():
        outputs = model(batch_frames)  # (max_frames, num_classes)
        outputs = torch.mean(outputs, dim=0)  # Average across frames
        pred_idx = torch.argmax(outputs).item()

    return classes[pred_idx]

if __name__ == "__main__":
    # Define your class list
    classes = ['walking', 'running', 'no_motion']

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FrameCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))

    # Specify the input video path here (replace with actual video path)
    input_video = r"your custom path"

    try:
        predicted_class = predict_motion(model, device, classes, input_video)
        print(f"Predicted motion for the video is: {predicted_class}")
    except Exception as e:
        print(f"Error during prediction: {e}")
