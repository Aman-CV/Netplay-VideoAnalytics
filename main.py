from fontTools.cffLib import encodeNumber

from trackers import ShuttleTracker, PlayerTracker
from constants import Directories, BadmintonCourtDimensions
from utils import (save_video,
                   read_video,
                   draw_frame_number,
)
from court_detector import CourtKeyPointsDetector
from hitframe_detector import HitFrameDetector
from rally import Rally
import pandas as pd


def main():
    _DIR = Directories()
    _DIMENSIONS = BadmintonCourtDimensions()
    real_court_keypoints = _DIMENSIONS.get_dimension_coordinates()

    # Get court keypoints
    court_keypoints_detector = CourtKeyPointsDetector()
    court_keypoints = court_keypoints_detector.get_court_keypoints()

    input_video_path = "{x}Test/match3/video/1_02_00.mp4".format(x=_DIR.INPUT_DIR)
    output_video_path = "{x}output.mp4".format(x=_DIR.OUTPUT_DIR)
    video_frames = read_video(input_video_path)

    # Get Player positions
    player_tracker = PlayerTracker("{x}yolo11x.pt".format(x=_DIR.MODEL_DIR))
    player_detections = player_tracker.detect_frames(video_frames,
                                                     True,
                                                     "{x}player_detections.pkl".format(x=_DIR.STUBS_DIR))
    player_detections = player_tracker.choose_and_filter_players(court_keypoints_detector.get_center(), player_detections)


    # Get shuttle positions
    shuttle_tracker = ShuttleTracker("{x}wasb_badminton_best.pth.tar".format(x=_DIR.MODEL_DIR))
    shuttle_detections = shuttle_tracker.get_ball_positions(input_video_path,
                                                            True,
                                                            "{x}shuttle_detections.pkl".format(x=_DIR.STUBS_DIR))




    # interpolate the shuttle position
    shuttle_detections = shuttle_tracker.interpolate_shuttle_position(shuttle_detections)

    # Get hitter frame
    hitframe_detector = HitFrameDetector(f"{_DIR.MODEL_DIR}x3d_m_best_50percentdata.pth")
    hitframe_detections = hitframe_detector.get_hiframes(video_frames, True, f"{_DIR.STUBS_DIR}hitframe_detections.pkl")
    df = pd.DataFrame({"marks" : hitframe_detections})
    df.to_csv(f"{_DIR.OUTPUT_DIR}marks.csv")
    real_hit_frame = [16, 79, 97, 119, 152, 179, 212, 237, 269, 300, 337, 366, 398, 427, 452, 476, 501, 529, 564, 582, 617, 637]
    rhf = [((i + 1) % 2, hit) for i, hit in enumerate(real_hit_frame)]
    print(rhf, len(real_hit_frame))
    print(hitframe_detections, len(hitframe_detections))

    # Rally
    rally_seq = Rally(player_detections, shuttle_detections, hitframe_detections, court_keypoints)

    # draw over video_frames
    shuttle_tracker.draw_circle(video_frames, shuttle_detections, True)
    player_tracker.draw_bboxes(video_frames, player_detections)
    court_keypoints_detector.draw_keypoints_on_video(video_frames)
    hitframe_detector.mark_hitframes(hitframe_detections, video_frames)
    # mark frame
    draw_frame_number(video_frames)

    # Save Video
    save_video(video_frames, output_video_path)




if __name__ == "__main__":
    main()