import os
from email.encoders import encode_noop

from constants import Directories
import random

_DIR = Directories()

video_dir = os.fsdecode(_DIR.VIDEO_CLIPS_DIR + "val/None/")
for i, file in enumerate(os.listdir(video_dir)):
    continue
print(i)
# random.seed(42)
# Define the range of integers
# start = 0
# end = 393
# random_integers = random.sample(range(start, end + 1), 393 - 75)
#
# ctr = 0
# for i, file in enumerate(os.listdir(video_dir)):
#     if i in random_integers:
#         os.remove(_DIR.VIDEO_CLIPS_DIR + "val/None/" + file)
#         ctr += 1
#
# print("file deleted , ", ctr)






