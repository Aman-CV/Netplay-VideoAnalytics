import os
from constants import Directories
from utils import read_video
import pandas as pd
import cv2
from PIL import Image
import random
_DIR = Directories()

def main():
    pass
random.seed(1)
if __name__ == "__main__":
    for f in os.listdir(r"C:\Users\AmanGautam\PycharmProjects\BadmintonCoachAI\data_preparation\checkpoint"):
        rand_num = random.choice([1, 2, 3, 4, 5])
        match_ = f[:-12]
        no_ = f[-11:-4]
        print(match_[5:], no_, rand_num)
        _name = f"{match_}_{no_}"

        input_video_frames = read_video("{x}Professional/{match}/video/{no}.mp4".format(x=_DIR.INPUT_DIR, match=match_, no=no_))
        csv_path = "{x}Professional/{match}/csv/{no}_ball.csv".format(x=_DIR.INPUT_DIR, match=match_, no=no_)
        if rand_num != 5:
            image_dir = f"{_DIR.INPUT_DIR}/shuttlecock_dataset/images/train/"
            label_dir = f"{_DIR.INPUT_DIR}/shuttlecock_dataset/labels/train/"
        else:
            image_dir = f"{_DIR.INPUT_DIR}/shuttlecock_dataset/images/val/"
            label_dir = f"{_DIR.INPUT_DIR}/shuttlecock_dataset/labels/val/"
        df = pd.read_csv(csv_path)

        for frame, (i, data) in zip(input_video_frames, df.iterrows()):
            fno, visibility, x, y = data
            img_h, img_w, ch = frame.shape
            image_name = _name + str(i) + ".jpg"
            label_name = _name + str(i) + ".txt"
            cv2.imwrite(image_dir + image_name, frame)
            box_w, box_h = 20 / img_w, 20 / img_h
            x = float(x) / img_w
            y = float(y) / img_h
            with open(label_dir + label_name, 'w') as f:
                if visibility != 0: f.write(f"0 {x:.6f} {y:.6f} {box_w:.6f} {box_h:.6f}\n")


