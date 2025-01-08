from ultralytics import YOLO
from trackers import ShuttleTracker
from constants import Directories
from utils import save_video, read_video, get_meta_data

def main():
    _DIR = Directories()
    input_video_path = "{x}Test/match3/video/1_02_00.mp4".format(x=_DIR.INPUT_DIR)
    output_video_path = "{x}output.mp4".format(x=_DIR.OUTPUT_DIR)
    video_frames = read_video(input_video_path)


    shuttle_tracker = ShuttleTracker("{x}wasb_badminton_best.pth.tar".format(x=_DIR.MODEL_DIR))
    shuttle_detections = shuttle_tracker.get_ball_positions(input_video_path,
                                                            True,
                                                            "{x}shuttle_detections.pkl".format(x=_DIR.STUBS_DIR))


    # draw over video_frames
    shuttle_tracker.draw_circle(video_frames, shuttle_detections)

    # Save Video
    save_video(video_frames, output_video_path)

if __name__ == "__main__":
    main()