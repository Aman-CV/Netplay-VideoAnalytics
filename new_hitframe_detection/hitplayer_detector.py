from pytorchvideo.models.hub import x3d_l, x3d_m
from torchvision.io import read_video
import torch.nn as nn
import torch
from constants import Directories
from utils import VideoTransform, save_video, draw_frame_number
import cv2
import pickle
from utils import expand_bbox
from utils import read_video as rv
import numpy as np
from trackers import PlayerTracker
import torchvision
_DIR = Directories()

class HitFrameDetector:
    def __init__(self, model_path_a, model_path_b):
        self.video_transform = VideoTransform(num_frames=16, resize=(214, 214))
        self.model_a = x3d_m(pretrained=False)
        self.num_classes = 2
        self.model_a.blocks[-1].proj = nn.Linear(in_features=2048, out_features=self.num_classes)
        c = torch.load(model_path_a)
        try:
            self.model_a.load_state_dict(c["model_state_dict"])
        except KeyError as e:
            self.model_a.load_state_dict(c)
        self.model_a.cuda()
        self.model_a.eval()  # Set the model to evaluation mode
        self.model_b = x3d_m(pretrained=False)
        self.num_classes = 2
        self.model_b.blocks[-1].proj = nn.Linear(in_features=2048, out_features=self.num_classes)
        c = torch.load(model_path_b)
        try:
            self.model_b.load_state_dict(c["model_state_dict"])
        except KeyError as e:
            self.model_b.load_state_dict(c)
        self.model_b.cuda()
        self.model_b.eval()

        self.model_x = torchvision.models.video.r3d_18(pretrained=False)
        self.model_x.fc = nn.Linear(self.model_x.fc.in_features, 2)  # Assuming 2 classes
        self.model_x.load_state_dict(torch.load("C:/Users/AmanGautam/Downloads/random_r18_epoch4.pth", map_location='cpu'))  # Update path
        self.model_x.eval()

    def inference(self, video_path):
        video, _, _ = read_video(video_path, pts_unit='sec')
        video = self.video_transform(video)
        video = video.unsqueeze(0).cuda()  # Add batch dimension and move to GPU
        video = video.permute(0, 2, 1, 3, 4)  # [batch_size, channels, num_frames, height, width]
        with torch.no_grad():
            outputs = self.model_b(video)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.item()  # Return the predicted class index


    def inference_from_frames(self, video_frames, player_detection_clip):
        cropped_video_frames = []
        for video_frame, player_dict in zip(video_frames, player_detection_clip):
            cvf = []
            i = 0
            for track_id, bbox in player_dict.items():
                if i == 2: break
                i += 1
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cropped_frame = video_frame[y1:y2, x1:x2]
                cropped_frame = cv2.resize(cropped_frame, (214, 214))
                cvf.append(cropped_frame)
            cropped_video_frames.append(cvf)
        cropped_video_frames = np.array(cropped_video_frames)
        number_of_detections = cropped_video_frames.shape[1]
        preds = []


        for index in range(number_of_detections):
            frames = cropped_video_frames[:, index, :, :, :]
            video = torch.stack([torch.from_numpy(f) for f in frames])
            video = self.video_transform(video)
            video = video.unsqueeze(0).cuda()  # Add batch dimension and move to GPU
            video = video.permute(0, 2, 1, 3, 4)  # [batch_size, channels, num_frames, height, width]
            with torch.no_grad():
                if index == 0:
                    outputs = self.model_a(video)
                else: outputs = self.model_b(video)
                predictions = torch.argmax(outputs, dim=1)
            preds.append(predictions.item())
        print(preds)
        return preds  # Return the predicted class index

    def get_hiframes(self, video_frames, player_detections, read_from_stub=False, stub_path=None):
        inference_output = []
        sliding_window = 8
        stride = 5
        for i in range(0, len(video_frames), stride):
            if i + sliding_window > len(video_frames): break
            clip = video_frames[i: i + sliding_window]
            player_detections_clip = player_detections[i: i + sliding_window]
            res = self.inference_from_frames(clip, player_detections_clip)
            flag = 2
            if res == [0, 1]:
                flag = 0
            elif res == [1, 0]:
                flag = 1
            inference_output.append([flag, i])
        refined_inference = [(-1, -1)]
        # for i in range(0, len(inference_output), stride):
        #     if i + sliding_window > len(inference_output): break
        #     clip = inference_output[i: i + sliding_window]
        #     a, b, none_ = clip.count(0), clip.count(1), clip.count(2)
        #     if a > 2:
        #         if refined_inference[-1][0] != 0: refined_inference.append([0, i])
        #     elif b > 2:
        #         if refined_inference[-1][0] != 1: refined_inference.append([1, i])
        # print(refined_inference)
        print(inference_output)
        for p, fr in inference_output:
            if p == 0:
                if refined_inference[-1][0] != 0:
                    refined_inference.append([0, fr])
            elif p == 1:
                if refined_inference[-1][0] != 1:
                    refined_inference.append([1, fr])
        inference_output = refined_inference[1:]

        return inference_output

    @staticmethod
    def mark_hitframes(hitframe_detections, video_frame):
        for player, frame_no in hitframe_detections:
            # Split the image into its color channels
            blue, green, red = cv2.split(video_frame[frame_no])

            # Increase the intensity of the green channel
            green = cv2.add(green, 50)

            # Ensure the values stay within the valid range [0, 255]
            green = np.clip(green, 0, 255)
            video_frame[frame_no] = cv2.merge([blue, green, red])


if __name__ == "__main__":

    input_video_path = "C:/Users/AmanGautam/PycharmProjects/BadmintonCoachAI/input_videos/Test/match3/video/1_02_00.mp4"

    hfd = HitFrameDetector("C:/Users/AmanGautam/Downloads/best.pth", _DIR.MODEL_DIR + "last_5_B.pth")


    player_tracker = PlayerTracker("{x}yolo11x.pt".format(x=_DIR.MODEL_DIR))
    vf = rv(input_video_path)
    pd = player_tracker.detect_frames(vf,
                         True,
                                    "{x}player_detections_hitplayer_detector.pkl".format(x=_DIR.STUBS_DIR))

    with open("../court_detector/keycheckpoint/court_points4_5avg.pkl", 'rb') as f:
        court_points = pickle.load(f)
    kpts = court_points[1]
    avg_x, avg_y = map(lambda v: sum(v) / len(kpts), zip(*kpts))
    kpts.append((avg_x, avg_y))
    pd = player_tracker.choose_and_filter_players([640, 460], pd)
    # pd = player_tracker.choose_and_filter_players([640, 460], pd)
    real_hit_frame = [16, 79, 97, 119, 152, 179, 212, 237, 269, 300, 337, 366, 398, 427, 452, 476, 501, 529, 564, 582, 617, 637]
    rhf = [((i + 1) % 2, hit) for i, hit in enumerate(real_hit_frame)]
    print(rhf, len(real_hit_frame))
    f = hfd.get_hiframes(vf, pd)
    player_tracker.draw_bboxes(vf, pd)
    draw_frame_number(vf)
    save_video(vf, "this.mp4")
    print(f, len(f))
