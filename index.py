from trackers import ShuttleTracker, PlayerTracker
from constants import Directories, BadmintonCourtDimensions
from utils import (save_video,
                   read_video,
                   draw_frame_number,
                   get_meta_data,
                   Transformer
)
from court_detector import CourtKeyPointsDetector
from new_hitframe_detection import HitFrameDetector
from rally import Rally
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
    input_video_path = "{x}{y}".format(x=_DIR.INPUT_DIR, y=name)
    print(input_video_path)
    if "/" in name:
        name  = name.split("/")[-1].split(".")[0]
    #input_video_path = "C:/Users/AmanGautam/PycharmProjects/BadmintonCoachAI/input_videos/Test/match3/video/1_02_00.mp4"
    player_from_stub, shuttle_from_stub, hitframe_from_stub = True, True, False
    output_video_path = "{x}op_{y}.mp4".format(x=_DIR.OUTPUT_DIR, y=name)
    fps, width, height = get_meta_data(input_video_path)
    video_frames = read_video(input_video_path)
    print(video_frames)
    print("len", len(video_frames))
    # Get Player positions
    player_tracker = PlayerTracker("{x}yolo11x.pt".format(x=_DIR.MODEL_DIR))
    player_detections = player_tracker.detect_frames(video_frames,
                                                  player_from_stub,
                                                  "{x}player_detections_{n}.pkl".format(x=_DIR.STUBS_DIR, n=name))
    player_detections = player_tracker.choose_and_filter_players(court_keypoints_detector.get_center(), player_detections)
    print(court_keypoints_detector.get_center())

    # Get shuttle positions
    shuttle_tracker = ShuttleTracker("wasb")
    shuttle_detections = shuttle_tracker.get_ball_positions(input_video_path,
                                                            shuttle_from_stub,
                                                            "{x}shuttle_detections_{y}.pkl".format(x=_DIR.STUBS_DIR,y=name))




    # interpolate the shuttle position
    shuttle_detections = shuttle_tracker.detect_outofview(shuttle_detections)
    shuttle_detections = shuttle_tracker.interpolate_shuttle_position(shuttle_detections)
    import numpy as np
    np_shuttle_detections = np.array(shuttle_detections, dtype=np.float32)
    shuttle_detections_df = pd.DataFrame(np_shuttle_detections)
    shuttle_detections_df = shuttle_detections_df.bfill(axis=0)
    shuttle_detections_df.to_csv("this.csv", index=False)
   # save_video(clip_hitframes(video_frames, get_hitframes(np_shuttle_detections)),"{x}REFop_{y}.mp4".format(x=_DIR.OUTPUT_DIR, y=name))
    # Get hitter frame
    hitframe_detector = HitFrameDetector(f"{_DIR.MODEL_DIR}last_3_A.pth", f"{_DIR.MODEL_DIR}last_5_B.pth")
    hitframe_detections = hitframe_detector.get_hiframes(video_frames, player_detections, hitframe_from_stub, f"{_DIR.STUBS_DIR}hitframe_detections_{name}.pkl")
    # df = pd.DataFrame({"marks" : hitframe_detections})
    # df.to_csv(f"{_DIR.OUTPUT_DIR}marks.csv")
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




if __name__ == "__main__":
    l = ["Test/match3/video/1_02_00.mp4"]
    for i in l:
        main(i)