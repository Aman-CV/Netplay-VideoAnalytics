from trackers import ShuttleTracker, PlayerTracker
from constants import Directories, BadmintonCourtDimensions
from utils import (save_video,
                   read_video,
                   draw_frame_number,
                   get_meta_data,
                   Transformer,
                   get_foot_position,
                    measure_distance
)
from court_detector import CourtKeyPointsDetector
from hitframe_detector import HitFrameDetector
from rally import Rally
import pandas as pd
from constants import DataParam
import os
import pickle



def main(input_video_path, _id):
    _DIR = Directories()
    _DIMENSIONS = BadmintonCourtDimensions()
    _DATA_PARAM = DataParam()
    real_court_keypoints = _DIMENSIONS.get_dimension_coordinates()
    # Get court keypoints
    court_keypoints_detector = CourtKeyPointsDetector()
    court_keypoints = court_keypoints_detector.get_court_keypoints()

    # input_video_path = "{x}Test/match2/video/1_03_03.mp4".format(x=_DIR.INPUT_DIR)
    # input_video_path  = "C:/Users/AmanGautam/PycharmProjects/BadmintonCoachAI/input_videos/Professional/match1/video/1_07_06.mp4"
    player_from_stub, shuttle_from_stub, hitframe_from_stub = False, False, False
    output_video_path = "{x}output.mp4".format(x=_DIR.OUTPUT_DIR)
    fps, width, height = get_meta_data(input_video_path)
    video_frames = read_video(input_video_path)

    # Get Player positions
    player_tracker = PlayerTracker("{x}yolo11x.pt".format(x=_DIR.MODEL_DIR))
    player_detections = player_tracker.detect_frames(video_frames,
                                                     player_from_stub,
                                                     "{x}player_detections.pkl".format(x=_DIR.STUBS_DIR))
    with open("court_detector/keycheckpoint/court_points4_5avg.pkl", 'rb') as f:
        court_points = pickle.load(f)
    kpts = court_points[_id - 1]
    avg_x, avg_y = map(lambda v: sum(v) / len(kpts), zip(*kpts))
    kpts.append((avg_x, avg_y))
    player_detections = player_tracker.choose_and_filter_players((avg_x, avg_y + 100), player_detections)


    # Get shuttle positions
    # shuttle_tracker = ShuttleTracker("{x}wasb_badminton_best.pth.tar".format(x=_DIR.MODEL_DIR))
    # shuttle_detections = shuttle_tracker.get_ball_positions(input_video_path,
    #                                                        shuttle_from_stub,
    #                                                        "{x}shuttle_detections_p1.pkl".format(x=_DIR.STUBS_DIR))




    # interpolate the shuttle position
    # shuttle_detections = shuttle_tracker.interpolate_shuttle_position(shuttle_detections)

    # Get hitter frame
    # hitframe_detector = HitFrameDetector(f"{_DIR.MODEL_DIR}large_x3d_m_3class_best_100per.pth")
    # hitframe_detections = hitframe_detector.get_hiframes(video_frames, hitframe_from_stub, f"{_DIR.STUBS_DIR}hitframe_detections.pkl")
    # print(hitframe_detections)

    # df = pd.DataFrame({"marks" : hitframe_detections})
    # df.to_csv(f"{_DIR.OUTPUT_DIR}marks.csv")
    # real_hit_frame = [16, 79, 97, 119, 152, 179, 212, 237, 269, 300, 337, 366, 398, 427, 452, 476, 501, 529, 564, 582, 617, 637]
    # rhf = [((i + 1) % 2, hit) for i, hit in enumerate(real_hit_frame)]
    # print(rhf, len(real_hit_frame))
    # print(hitframe_detections, len(hitframe_detections))

    # Rally
    # rally_seq = Rally(player_detections, shuttle_detections, hitframe_detections, court_keypoints, real_court_keypoints, fps)

    # draw over video_frames
    # shuttle_tracker.draw_circle(video_frames, shuttle_detections, True)
    player_tracker.draw_bboxes(video_frames, player_detections)
    flat_list = [item for tup in kpts for item in tup]
    court_keypoints_detector.set_keypoints(flat_list)
    court_keypoints_detector.draw_keypoints_on_video(video_frames)
    # hitframe_detector.mark_hitframes(hitframe_detections, video_frames)
    # mark frame
    draw_frame_number(video_frames)
    # rally_seq.draw_statistics_on_video(video_frames, _DATA_PARAM.speed)
    # Save Video
    save_video(video_frames, output_video_path)




if __name__ == "__main__":
    base_dir = "C:/Users/AmanGautam/PycharmProjects/BadmintonCoachAI/input_videos/Professional/"
    for i in range(3, 24):
        match_ = base_dir + "match" + str(i) + "/video/"
        for file in os.listdir(match_):
            main(match_ + str(file), i)
            input("prompt kk: ")