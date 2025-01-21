import cv2
from torchvision import transforms
import torch

def get_meta_data(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, width, height

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def draw_frame_number(video_frames):
    for i, frame in enumerate(video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def get_sample_frame(video_frames):
    cv2.imwrite("sample.png", video_frames[1])


class VideoTransform:
    def __init__(self, num_frames=8, resize=(160, 160)):
        self.num_frames = num_frames
        self.resize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, video):
        T, H, W, C = video.shape

        # Handle frame sampling or padding
        if T > self.num_frames:
            indices = torch.linspace(0, T - 1, self.num_frames).long()
            video = video[indices]
        elif T < self.num_frames:
            pad = self.num_frames - T
            padding = video[-1:].repeat(pad, 1, 1, 1)
            video = torch.cat([video, padding], dim=0)

        # Transform each frame
        transformed_frames = []
        for frame in video:
            frame = frame.permute(2, 0, 1)  # Convert to [C, H, W] format
            frame = self.resize(frame)  # Apply resize
            frame = self.normalize(frame)  # Normalize the frame
            transformed_frames.append(frame)

        video = torch.stack(transformed_frames)  # [T, C, H, W]
        return video
