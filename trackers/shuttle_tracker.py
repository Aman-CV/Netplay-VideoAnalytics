from inference_scripts.wasb_inference import run_inference as wasb_inference
from inference_scripts.tracknetv2_inference import run_inference as tracknetv2_inference
import cv2
import pickle
import numpy as np
import pandas as pd
from constants import  Directories
DIR_ = Directories()
# Map models to their corresponding inference functions, I am using only wasb, if needed more we can add here check out WASB gitHub repo
MODEL_INFERENCE_MAP = {
    "wasb": wasb_inference,
    "tnet": tracknetv2_inference
}
MODEL_PATH_MAP = {
    "wasb": "wasb_badminton_best.pth.tar",
    "tnet": "tracknetv2_badminton_best.pth.tar"
}
class ShuttleTracker:
    def __init__(self, model):
        self.model_path = DIR_.MODEL_DIR + MODEL_PATH_MAP[model]
        self.__inference_function = MODEL_INFERENCE_MAP[model]
        self.__is_interpolated = False

    def get_hit_frame(self, shuttle_detections):
        if not self.__is_interpolated:
            print("LOG : dataset contain NaN values")
            return shuttle_detections
        DIRECTIONS = {0: "Steady", 1: "Upper Court", 2: "Bottom Court"}
        pass

    def get_ball_positions(self, input_path, read_from_stub=False, stub_path=None):
        if not read_from_stub:
            ball_detections = self.__inference_function(weights=self.model_path, input_path=input_path)
            if stub_path:
                with open(stub_path, 'wb') as f:
                    # noinspection PyTypeChecker
                    pickle.dump(ball_detections, f)
        else:
            if not stub_path:
                raise Exception("Stub path missing")
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
        return ball_detections

    def draw_circle(self, video_frames, shuttle_detections, is_interpolated=True):
        for frame, shuttle_detection in zip(video_frames, shuttle_detections):
            frame_num, is_detected, x, y, cnf = shuttle_detection
            if is_detected or self.__is_interpolated:
                cv2.circle(frame,[int(x), int(y)], radius=5, color=(255, 255, 255), thickness=2)
            #cv2.putText(frame, f"Confidence of shuttle detection: {round(cnf, 2)}", (800, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def interpolate_shuttle_position(self, shuttle_detections):
        np_shuttle_detections = np.array(shuttle_detections, dtype=np.float32)
        np_shuttle_detections[np_shuttle_detections[:, 1] == 0, 2:4] = np.nan
        shuttle_detections_df = pd.DataFrame(np_shuttle_detections)
        shuttle_detections_df = shuttle_detections_df.bfill(axis=0)
        shuttle_detections_df.to_csv("dfo.csv", index=False)
        shuttle_detections_df[2] = shuttle_detections_df[2].interpolate()
        shuttle_detections_df[3] = shuttle_detections_df[3].interpolate()
        np_shuttle_detections = shuttle_detections_df.to_numpy()
        shuttle_detections_df.to_csv("dfm.csv", index=False)
        self.__is_interpolated = True
        return np_shuttle_detections.tolist()

    def detect_outofview(self, shuttle_detections):
        flag = False
        for i in range(1, len(shuttle_detections)):
            frame_num, is_detected, x, y, cnf = shuttle_detections[i]
            _, _, _, y_prev, _ = shuttle_detections[i - 1]
            if not is_detected and y_prev < 20:
                is_detected = True
                y = y_prev
        return shuttle_detections




