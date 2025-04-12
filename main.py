import cv2
import matplotlib.pyplot as plt

from trackers import ShuttleTracker, PlayerTracker
from constants import Directories, BadmintonCourtDimensions
from utils import (save_video,
                   read_video,
                   draw_frame_number,
                   get_meta_data,
                   Transformer
)
from court_detector import CourtKeyPointsDetector
from hitframe_detector import HitFrameDetector
from rally import Rally
import numpy as np
import pandas as pd
from constants import DataParam
# from new_hitframe_detection.m_hitframe_detector import get_hitframes, clip_hitframes

def main(inp = None):
    _DIR = Directories()
    _DIMENSIONS = BadmintonCourtDimensions()
    _DATA_PARAM = DataParam()
    real_court_keypoints = _DIMENSIONS.get_dimension_coordinates()
    # Get court keypoints
    court_keypoints_detector = CourtKeyPointsDetector()
    court_keypoints = court_keypoints_detector.get_court_keypoints()
    name = inp
    #input_video_path = "{x}{y}.mp4".format(x=_DIR.INPUT_DIR, y=name)
    input_video_path = "C:/Users/AmanGautam/PycharmProjects/BadmintonCoachAI/input_videos/p1.mp4"
    player_from_stub, shuttle_from_stub, hitframe_from_stub = False , False, False
    output_video_path = "{x}op_{y}.mp4".format(x=_DIR.OUTPUT_DIR, y=name)
    fps, width, height = get_meta_data(input_video_path)
    video_frames = read_video(input_video_path)
    print(video_frames)
    print("len", len(video_frames))
    # Get Player positions
    player_tracker = PlayerTracker("{x}yolov8x.pt".format(x=_DIR.MODEL_DIR))
    player_detections = player_tracker.detect_frames(video_frames,
                                                  player_from_stub,
                                                 "{x}player_detections.pkl".format(x=_DIR.STUBS_DIR))
    player_detections = player_tracker.choose_and_filter_players(court_keypoints_detector.get_center(), player_detections)


    # Get shuttle positions
    shuttle_tracker = ShuttleTracker("wasb")
    shuttle_detections = shuttle_tracker.get_ball_positions(input_video_path,
                                                            shuttle_from_stub,
                                                            "{x}shuttle_detections_{y}.pkl".format(x=_DIR.STUBS_DIR,y=name))




    # interpolate the shuttle position
    shuttle_detections = shuttle_tracker.detect_outofview(shuttle_detections)
    shuttle_detections = shuttle_tracker.interpolate_shuttle_position(shuttle_detections)
    # np_shuttle_detections = np.array(shuttle_detections, dtype=np.float32)
    # shuttle_detections_df = pd.DataFrame(np_shuttle_detections)
    # shuttle_detections_df = shuttle_detections_df.bfill(axis=0)
    # shuttle_detections_df.to_csv("rally_dat.csv", index=False)
    # save_video(clip_hitframes(video_frames, get_hitframes(np_shuttle_detections)),"{x}REFop_{y}.mp4".format(x=_DIR.OUTPUT_DIR, y=name))
    # Get hitter frame
    hitframe_detector = HitFrameDetector(f"{_DIR.MODEL_DIR}large_x3d_m_3class_best_100per.pth")
    hitframe_detections, prp = hitframe_detector.get_hiframes(video_frames, hitframe_from_stub, f"{_DIR.STUBS_DIR}hitframe_detections.pkl")
    print(hitframe_detections)
    df = pd.DataFrame({"marks" : hitframe_detections})
    dfn = pd.DataFrame({"prob": prp})
    x = np.arange(0, len(video_frames), 1)
    from scipy.ndimage import gaussian_filter1d
    smoothed_probs = gaussian_filter1d(prp, sigma=2)
    plt.plot(x, smoothed_probs)
    real_hit_frame = [16, 79, 97, 119, 152, 179, 212, 237, 269, 300, 337, 366, 398, 427, 452, 476, 501, 529, 564, 582, 617, 637]
    rhf = [((i + 1) % 2, hit) for i, hit in enumerate(real_hit_frame)]
    print(rhf, len(real_hit_frame))
    print(hitframe_detections, len(hitframe_detections))

    # Rally
    # rally_seq = Rally(player_detections, shuttle_detections, hitframe_detections, court_keypoints, real_court_keypoints, fps)

    # draw over video_frames
    shuttle_tracker.draw_circle(video_frames, shuttle_detections, True)
    player_tracker.draw_bboxes(video_frames, player_detections)
    # court_keypoints_detector.draw_keypoints_on_video(video_frames)
    hitframe_detector.mark_hitframes(hitframe_detections, video_frames)
    # mark frame
    draw_frame_number(video_frames)
    # rally_seq.draw_statistics_on_video(video_frames, _DATA_PARAM.speed)
    # Save Video
    save_video(video_frames, output_video_path, fps)
    plt.show()



if __name__ == "__main__":
    #frames = read_video("C:/Users/AmanGautam/PycharmProjects/BadmintonCoachAI/output_videos/op_['p1'].mp4")
    #cv2.imwrite("sample.jpg", frames[20])
    l = ["p1"],
    for i in l:
       main(i)