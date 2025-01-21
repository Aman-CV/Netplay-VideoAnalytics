import cv2
import numpy as np
# import math
# from court_line_detector import (CourtLineDetector)
# from utils import (read_video,
#                    save_video)


class Transformer:
    def __init__(self, keypoints, destination_coordinates):
        #self.corner_points = [0, 1, 2, 3]
        self.img_points = np.column_stack((keypoints[0::2], keypoints[1::2]))
        self.real_world_points = np.column_stack((destination_coordinates[0::2], destination_coordinates[1::2]))
        self._H, _ = cv2.findHomography(self.img_points, self.real_world_points)


    def get_homographic_tensor(self):
        return self._H

    def get_real_life(self, pixel_coordinate):
        fake_life = np.array([pixel_coordinate[0], pixel_coordinate[1], 1], dtype=np.float32)[:, None]
        real_life = self._H @ fake_life
        real_life = real_life / real_life[2, 0]
        return [real_life[0], real_life[1]]



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

if __name__ == "__main__":
    pass
    # input_videos_path = "../input_videos/input_video.mp4"
    # video_frames = read_video(input_videos_path)
    # court_model_path = "../models/A12keypoints_model.pth"
    # court_line_detector = CourtLineDetector(court_model_path)
    # court_keypoints = court_line_detector.predict(video_frames[0])
    #
    # T = Transformer(video_frames[0], court_keypoints)


