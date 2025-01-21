import cv2
from constants import Directories
from utils import (read_video, save_video, get_meta_data)
import pandas as pd
import pickle
import os

_DIR = Directories()

def show_marked_frame(frame, ip):
    fcopy = frame.copy()
    cv2.putText(fcopy, ip, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Video Frames", fcopy)

def video_labeller(video_frames, read_from_check_point=False, checkpoints_path=None):
    if read_from_check_point and checkpoints_path:
        with open(checkpoints_path, 'rb') as f:
            frame_inputs = pickle.load(f)
            return frame_inputs
    frame_inputs = ['n'] * len(video_frames)
    current_frame = 0
    cv2.namedWindow("Video Frames")
    while True:
        if current_frame == len(video_frames): current_frame -= 1
        show_marked_frame(video_frames[current_frame], frame_inputs[current_frame])
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            current_frame -= int(current_frame > 0)
            continue
        elif key == ord('a'):
            frame_inputs[current_frame] = 'a'
            current_frame += int(current_frame < len(video_frames))
            continue
        elif key == ord('b'):
            frame_inputs[current_frame] = 'b'
            current_frame += int(current_frame < len(video_frames))
            continue
        elif key == ord('n'):
            frame_inputs[current_frame] = 'n'
            current_frame += int(current_frame < len(video_frames))
            continue
        elif key == ord('p'):
            print("Exiting")
            break
        else:
            current_frame += int(current_frame < len(video_frames))
            continue
    if checkpoints_path:
        with open(checkpoints_path, "wb") as f:
            pickle.dump(frame_inputs, f)
    return frame_inputs


def create_data(video_frames, frame_inputs, sliding_window=8, base_name="", stride=5):
    clip_label = []
    for i in range(0, len(video_frames) - sliding_window, stride):
        if i + sliding_window > len(video_frames): break
        clip = video_frames[i: i + sliding_window]
        if 'a' in frame_inputs[i: i + sliding_window]:
            clip_path = _DIR.VIDEO_CLIPS_DIR + "val/" + "A/" + base_name + str(i) + ".mp4"
            clip_label.append((base_name + str(i), 'a'))
        elif 'b' in frame_inputs[i: i + sliding_window]:
            clip_path = _DIR.VIDEO_CLIPS_DIR + "val/" + "B/" + base_name + str(i) + ".mp4"
            clip_label.append((base_name + str(i), 'b'))
        else:
            clip_path = _DIR.VIDEO_CLIPS_DIR + "val/" + "None/" + base_name + str(i) + ".mp4"
            clip_label.append((base_name + str(i), 'n'))
        save_video(clip, clip_path)
    # df = pd.DataFrame(clip_label)
    # df.to_csv("train.csv", mode='a', index=False, header=False)



if __name__ == "__main__":
    no_ = "3_03_03"
    _name = f"match12_{no_}"
    input_video_frames = read_video("{x}Professional/match12/video/{no}.mp4".format(x=_DIR.INPUT_DIR, no=no_))
    frame_ip = video_labeller(input_video_frames, False, f"checkpoint/{_name}.pkl")
    create_data(input_video_frames, frame_ip, 8, _name)