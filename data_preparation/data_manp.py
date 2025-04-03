import os
from constants import Directories
import cv2
from utils import read_video, save_video
import pickle

input_video_path = "C:/Users/AmanGautam/PycharmProjects/BadmintonCoachAI/input_videos/Test/match3/video/1_02_00.mp4"
input_video_frames = read_video(input_video_path)
sliding_window = 8
stride = 5
for i in range(0, len(input_video_frames), stride):
    if i + sliding_window > len(input_video_frames): break
    clip = input_video_frames[i: i + sliding_window]
    save_video(clip, f"test/test_{i}.mp4")
# directory = "video_clips"
#
# for file in os.listdir("video_clips/train/A"):
#     cdata_path = "video_clips/train/A" + file.replace(".mp4", ".plk")
#     with open(cdata_path, 'rb') as f:
#         player_detections = pickle.load(f)
#     frames = read_video("video_clips/train/A" + file)
#     for frame, player_dict in zip(frames, player_detections):
#         bboxes = []
#         for track_id, bbox in player_dict.items():
#             if track_id == 2:
#                 continue
#             bboxes.append(bbox)
# Define the range of integers
# start = 0
# end = i
# random_integers = random.sample(range(start, end + 1), i - tosave)
#
# ctr = 0
# for i, file in enumerate(os.listdir(video_dir)):
#     if i in random_integers:
#         os.remove(_DIR.VIDEO_CLIPS_DIR + fpath + file)
#         ctr += 1
#
# print("file deleted , ", ctr)
#





