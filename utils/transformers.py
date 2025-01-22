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
        return [float(real_life[0, 0]), float(real_life[1, 0])]


if __name__ == "__main__":
    pass
    # input_videos_path = "../input_videos/input_video.mp4"
    # video_frames = read_video(input_videos_path)
    # court_model_path = "../models/A12keypoints_model.pth"
    # court_line_detector = CourtLineDetector(court_model_path)
    # court_keypoints = court_line_detector.predict(video_frames[0])
    #
    # T = Transformer(video_frames[0], court_keypoints)


