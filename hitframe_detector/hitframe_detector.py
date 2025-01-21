from pytorchvideo.models.hub import x3d_m
from torchvision.io import read_video
import torch.nn as nn
import torch
from constants import Directories
from utils import VideoTransform
import cv2
import pickle
import numpy as np
_DIR = Directories()

class HitFrameDetector:
    def __init__(self, model_path):
        self.video_transform = VideoTransform(num_frames=16, resize=(224, 224))
        self.model = x3d_m(pretrained=False)
        self.num_classes = 3
        self.model.blocks[-1].proj = nn.Linear(in_features=2048, out_features=self.num_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.cuda()
        self.model.eval()  # Set the model to evaluation mode

    def inference(self, video_path):
        video, _, _ = read_video(video_path, pts_unit='sec')
        video = self.video_transform(video)
        video = video.unsqueeze(0).cuda()  # Add batch dimension and move to GPU
        video = video.permute(0, 2, 1, 3, 4)  # [batch_size, channels, num_frames, height, width]
        with torch.no_grad():
            outputs = self.model(video)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.item()  # Return the predicted class index

    def inference_from_frames(self, video_frames):
        video = torch.stack([torch.from_numpy(f) for f in video_frames])
        video = self.video_transform(video)
        video = video.unsqueeze(0).cuda()  # Add batch dimension and move to GPU
        video = video.permute(0, 2, 1, 3, 4)  # [batch_size, channels, num_frames, height, width]
        with torch.no_grad():
            outputs = self.model(video)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.item()  # Return the predicted class index

    def get_hiframes(self, video_frames, read_from_stub=False, stub_path=None):
        inference_output = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                inference_output = pickle.load(f)
            return inference_output
        sliding_window = 8
        for i in range(0, len(video_frames) - sliding_window, 1):
            if i + sliding_window > len(video_frames): break
            clip = video_frames[i: i + sliding_window]
            inference_output.append(self.inference_from_frames(clip))
        riop = [(-1, -1)]
        for i in range(0, len(inference_output), 1):
            clip = inference_output[i: i + 5]
            a, b, n = clip.count(0), clip.count(1), clip.count(2)
            if a > 2:
                if riop[-1][0] != 0:
                    riop.append((0, i + 5))
                else:
                    riop[-1] = (0, i)
            elif b > 2:
                if riop[-1][0] != 1:
                    riop.append((1, i + 5))
                else:
                    riop[-1] = (1, i)
            else: continue
        inference_output = riop[1:]
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                # noinspection PyTypeChecker
                pickle.dump(inference_output, f)
        return inference_output

    @staticmethod
    def mark_hitframes(hitframe_detections, video_frame):
        for player, frame_no in hitframe_detections:
            # Split the image into its color channels
            blue, green, red = cv2.split(video_frame[frame_no])

            # Increase the intensity of the green channel
            green = cv2.add(green, 50)  # You can adjust this value for stronger/softer effect

            # Ensure the values stay within the valid range [0, 255]
            green = np.clip(green, 0, 255)
            video_frame[frame_no] = cv2.merge([blue, green, red])


if __name__ == "__main__":
    vid_path = f"{_DIR.VIDEO_CLIPS_DIR}train/A/match1_1_02_030.mp4"  # replace with the path to your video

    cap = cv2.VideoCapture(vid_path)
    frames = []
    num_frames = 8
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    cap.release()

    hitframe_detector = HitFrameDetector(f"{_DIR.MODEL_DIR}x3d_m_best_50percentdata.pth")
    predicted_class = hitframe_detector.inference(vid_path)
    predicted_class2 = hitframe_detector.inference_from_frames(frames)
    print(f"Predicted class: {predicted_class2}")
    print(f"Predicted class: {predicted_class}")

