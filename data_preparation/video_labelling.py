import cv2
from constants import Directories
from utils import (read_video, save_video, get_meta_data)
from os import listdir
import pandas as pd
import pickle
import os
import random
from trackers import PlayerTracker

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


def add_classes(frame_ip_):
    frame_inputs = frame_ip_.copy()
    start_class = "null"
    change_ = True

    for i in range(len(frame_inputs)):
        if (start_class =='b' and frame_inputs[i] == 'a') or (start_class == 'a' and frame_inputs[i] == 'b'):
            if change_: change_ = False
        if frame_inputs[i] == 'a' and change_:
            start_class = 'a'
            frame_inputs[i] = 'c'
        if frame_inputs[i] == 'b' and change_:
            start_class = 'b'
            frame_inputs[i] = 'd'


    return frame_inputs

def read_frame_input_from_checkpoints(path_to_checkpoint):
    with open(path_to_checkpoint, 'rb') as f:
        frame_ips = pickle.load(f)
        return frame_ips


def create_data_with_serve(video_frames, frame_inputs, sliding_window=16, base_name="", stride=16, pt=None, id_ = None):
    flag_folder = random.randint(1, 5)
    folder_type = "train/"
    if flag_folder == 5: folder_type = "val/"
    print(folder_type)

    for i in range(0, len(video_frames) - sliding_window, stride):
        if i + sliding_window > len(video_frames): break
        clip = video_frames[i: i + sliding_window]
        player_detections_clip = pt[i: i + sliding_window]

       # player_detections = pt.choose_and_filter_players([640, 360],
       #                                                  player_detections)
        if 'a' in frame_inputs[i: i + sliding_window]:
            clip_path = _DIR.VIDEO_CLIPS_DIR + folder_type + "A/" + base_name + str(i) + ".mp4"
            track_clip_path = "cropped_video_clips_pos/" + folder_type + "A/" + base_name + str(i) + ".pkl"
        elif 'b' in frame_inputs[i: i + sliding_window]:
            clip_path = _DIR.VIDEO_CLIPS_DIR + folder_type + "B/" + base_name + str(i) + ".mp4"
            track_clip_path = "cropped_video_clips_pos/" + folder_type + "B/" + base_name + str(i) + ".pkl"
        # elif 'c' in frame_inputs[i: i + sliding_window]:
        #     clip_path = _DIR.VIDEO_CLIPS_DIR + folder_type + "SA/" + base_name + str(i) + ".mp4"
        #     track_clip_path = "cropped_video_clips_pos/" + folder_type + "SA/" + base_name + str(i) + ".pkl"
        # elif 'd' in frame_inputs[i: i + sliding_window]:
        #     clip_path = _DIR.VIDEO_CLIPS_DIR + folder_type + "SB/" + base_name + str(i) + ".mp4"
        #     track_clip_path = "cropped_video_clips_pos/" + folder_type + "SB/" + base_name + str(i) + ".pkl"
        else:
            clip_path = _DIR.VIDEO_CLIPS_DIR + folder_type + "None/" + base_name + str(i) + ".mp4"
            track_clip_path = "cropped_video_clips_pos/" + folder_type + "None/" + base_name + str(i) + ".pkl"

        save_video(clip, clip_path)
        with open(track_clip_path, 'wb') as fk:
            # noinspection PyTypeChecker
            pickle.dump(player_detections_clip, fk)


def create_data(video_frames, frame_inputs, sliding_window=8, base_name="", stride=6):
    clip_label = []
    is_first_serve = True
    for i in range(0, len(video_frames) - sliding_window, stride):
        if i + sliding_window > len(video_frames): break
        clip = video_frames[i: i + sliding_window]
        if 'a' in frame_inputs[i: i + sliding_window]:
            clip_path = _DIR.VIDEO_CLIPS_DIR + "val/" + "A/" + base_name + str(i) + ".mp4"
            clip_label.append((base_name + str(i), 'a'))
        elif 'b' in frame_inputs[i: i + sliding_window]:
            clip_path = _DIR.VIDEO_CLIPS_DIR + "val/" +                  "B/" + base_name + str(i) + ".mp4"
            clip_label.append((base_name + str(i), 'b'))
        else:
            clip_path = _DIR.VIDEO_CLIPS_DIR + "val/" + "None/" + base_name + str(i) + ".mp4"
            flag = random.randint(1, 5)
            if flag == 3:
                clip_label.append((base_name + str(i), 'n'))
        save_video(clip, clip_path)
    # df = pd.DataFrame(clip_label)...
    # df.to_csv("train.csv", mode='a', index=False, header=False)


def correct_first_detect(frame_ip_):
    frame_inputs = frame_ip_.copy()
    if frame_inputs[0] != 'n':
        frame_inputs[3] = frame_inputs[0]
        frame_inputs[0] = 'n'
    return frame_inputs

if __name__ == "__main__":
    it = 0
    player_tracker = PlayerTracker("{x}yolo11x.pt".format(x=_DIR.MODEL_DIR))
    random.seed(1)
    for f in listdir(r"C:\Users\AmanGautam\PycharmProjects\BadmintonCoachAI\data_preparation\checkpoint"):
        it+=1
        match_ = f[:-12]
        no_ = f[-11:-4]
        print(match_[5:], no_, it)
        input_video_frames = read_video("{x}Professional/{match}/video/{no}.mp4".format(x=_DIR.INPUT_DIR, match=match_, no=no_))
        fp = read_frame_input_from_checkpoints("checkpoint/" + f)
        f_ac = correct_first_detect(fp)
        # f_ac = add_classes(fp)
        _name = f"{match_}_{no_}"
        player_detections = player_tracker.detect_frames(input_video_frames, True, stub_path=f"local_stub/{_name}.pkl")
        with open("../court_detector/keycheckpoint/court_points4_5avg.pkl", 'rb') as f:
            court_points = pickle.load(f)
        kpts = court_points[int(match_[5:]) - 1]
        avg_x, avg_y = map(lambda v: sum(v) / len(kpts), zip(*kpts))
        print(avg_x, avg_y)
        player_detections = player_tracker.choose_and_filter_players((avg_x, avg_y + 100), player_detections)
        create_data_with_serve(input_video_frames, f_ac, 16, _name, 16, pt=player_detections, id_=int(match_[5:]))
    # no_ = "1_01_00"
    # _name = f"match1_{no_}"
    # input_video_frames = read_video("{x}Professional/match1/video/{no}.mp4".format(x=_DIR.INPUT_DIR, no=no_))
    # frame_ip = video_labeller(input_video_frames, True, f"checkpoint/{_name}.pkl")
    # print(frame_ip)
    #
    # fp = add_classes(frame_ip)
    # print(frame_ip)
    # import csv
    # rows = zip(frame_ip, fp)
    # with open("this.csv", "w", newline='') as f:
    #     writer = csv.writer(f)
    #     for row in rows:+0
    #         writer.writerow(row)
    # create_data(input_video_frames, frame_ip, 8, _name)