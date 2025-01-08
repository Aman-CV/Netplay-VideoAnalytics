from inference_scripts.wasb_inference import run_inference as wasb_inference
import cv2
import pickle

# Map models to their corresponding inference functions, i am using only wasb, if needed more we can add here check out WASB github repo
MODEL_INFERENCE_MAP = {
    "wasb": wasb_inference
}

class ShuttleTracker:
    def __init__(self, model_path):
        self.model_path = model_path
        self.__inference_function = MODEL_INFERENCE_MAP["wasb"]

    def get_ball_positions(self, input_path, read_from_stub=False, stub_path=None):
        ball_detections = []
        if not read_from_stub:
            ball_detections = self.__inference_function(weights=self.model_path, input_path=input_path)
            if stub_path:
                with open(stub_path, 'wb') as f:
                    pickle.dump(ball_detections, f)
        else:
            if not stub_path:
                raise Exception("Stub path missing")
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
        return ball_detections

    def draw_circle(self, video_frames, shuttle_detections):
        for frame, shuttle_detection in zip(video_frames, shuttle_detections):
            frame_num, is_detected, x, y, cnf = shuttle_detection
            if is_detected and cnf > 20.:
                cv2.circle(frame,[int(x), int(y)], radius=5, color=(0, 255, 0), thickness=2)

    def interpolate_shuttle_position(self, shuttle_detections):
        pass


