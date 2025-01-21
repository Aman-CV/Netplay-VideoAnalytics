from utils import get_foot_position
import pandas as pd

class Rally:
    def __init__(self, player_detections, shuttle_detections, hitframe_detections, court_keypoints):
        # TODO: make this in tracker and make it more robust and shift it to player tracker will help in doubles
        y_centers = []
        for playerID, bbox in player_detections[0].items():
            x1, y1, x2, y2 = bbox
            y_centers.append([playerID, y1 / 2. + y2 / 2.])
        sorted(y_centers, key=lambda x: x[1]) # descending order sorting
        self.playerA_ID, self.playerB_ID = y_centers[0][0], y_centers[1][0]
        self.rally_data_table = []
        # shot, player who shot, position of shoter_x, position of shooter_y, frame no
        for i, hit_frame in enumerate(hitframe_detections):
            side, frame_no = hit_frame
            # TODO: Adjust this logic for double.
            player_who_took_shot = [self.playerA_ID, self.playerB_ID][side]
            bbox = player_detections[i][player_who_took_shot]
            position_of_player_who_took_shot = get_foot_position(bbox)
            self.rally_data_table.append([i, player_who_took_shot, position_of_player_who_took_shot[0], position_of_player_who_took_shot[1], frame_no])

        df = pd.DataFrame(self.rally_data_table, columns=['shot_no', 'player who shot', 'his position x', 'his position y', 'time stamp (frame no)'])
        df.to_csv("rally_data.csv")
