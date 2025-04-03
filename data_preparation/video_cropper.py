import cv2
from constants import Directories
from utils import (read_video, save_video, get_meta_data)
from os import listdir
import pandas as pd
import pickle
import os
import random

_DIR = Directories()

populate_dir = "cropped_video_clips_pos/train/A"
source_dir = "video_clips/train/A"
for i in os.listdir(source_dir):
    print(i)
