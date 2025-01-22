import numpy as np
from utils import get_foot_position
import pandas as pd
from utils import Transformer
from constants import DataParam
import cv2

def get_index(my_list, item):
    try:
        index = my_list.index(item)
        return index
    except ValueError:
        return -1

class Rally:
    def __init__(self, player_detections, shuttle_detections, hitframe_detections, court_keypoints, real_court_dimensions, fps):
        # TODO: make this in tracker and make it more robust and shift it to player tracker will help in doubles
        self._DATA_PARAM = DataParam()
        y_centers = []
        for playerID, bbox in player_detections[0].items():
            x1, y1, x2, y2 = bbox
            y_centers.append([playerID, y1 / 2. + y2 / 2.])
        y_centers.sort(key=lambda x: x[1]) # descending order sorting
        self.playerA_ID, self.playerB_ID = y_centers[0][0], y_centers[1][0]
        self.rally_data_table = []
        # shot, player who shot, position of shooter_x, position of shooter_y, frame no, transformed x, transformed y
        transform = Transformer(court_keypoints, real_court_dimensions)
        for i, hit_frame in enumerate(hitframe_detections):
            side, frame_no = hit_frame
            # TODO: Adjust this logic for double.
            player_who_took_shot = [self.playerA_ID, self.playerB_ID][side]
            bbox = player_detections[i][player_who_took_shot]
            shooter_position = get_foot_position(bbox)
            real_shooter_position = transform.get_real_life(shooter_position)
            distance = 0
            time = 0
            speed = 0
            if i != 0:
                prev_pos = self.rally_data_table[-1][self._DATA_PARAM.transformed_x: self._DATA_PARAM.transformed_y + 1]
                print(prev_pos, real_shooter_position, fps)
                distance = np.sqrt((real_shooter_position[0] - prev_pos[0]) ** 2 + (real_shooter_position[1] - prev_pos[1]) ** 2)
                time = (frame_no  - self.rally_data_table[-1][self._DATA_PARAM.frame_no]) / fps
                speed = distance / time
            self.rally_data_table.append([i, player_who_took_shot, shooter_position[0], shooter_position[1], frame_no, real_shooter_position[0], real_shooter_position[1], distance, time, speed])
        self.rally_data_table = np.array(self.rally_data_table)
        df = pd.DataFrame(self.rally_data_table, columns=self._DATA_PARAM.column_list)
        df.to_csv("rally_data.csv")


    def get_statistics(self):
        return self.rally_data_table

    def draw_statistics_on_video(self, video_frames, requested_param=9):
        speed = 0
        for i, frame in enumerate(video_frames):
            speeds = self.rally_data_table[self.rally_data_table[:, self._DATA_PARAM.frame_no] == i, self._DATA_PARAM.speed]
            if len(speeds) > 0:
                speed = speeds[0]
            cv2.putText(frame, f"Speed: {speed}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (87, 255, 60), 2)







