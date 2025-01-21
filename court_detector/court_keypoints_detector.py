import cv2

class CourtKeyPointsDetector:
    def __init__(self):
        self.__keypoints = [293, 659, 987, 659, 416, 305, 863, 305]
        self.__center = [sum(self.__keypoints[i] for i in range(0, len(self.__keypoints), 2)) / len(self.__keypoints[::2]),
                                sum(self.__keypoints[i] for i in range(1, len(self.__keypoints), 2)) / len(self.__keypoints[::2])]

    def get_court_keypoints(self):
        return self.__keypoints

    def get_center(self):
        return self.__center

    def draw_keypoints(self, image):
        # Plot keypoints on the image
        for i in range(0, len(self.__keypoints), 2):
            x = int(self.__keypoints[i])
            y = int(self.__keypoints[i + 1])
            cv2.putText(image, str(i // 2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    def draw_keypoints_on_video(self, video_frames):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame)
            output_video_frames.append(frame)
        return output_video_frames

