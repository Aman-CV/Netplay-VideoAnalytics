from torchaudio.io import play_audio
from ultralytics import YOLO
import pickle
import cv2
from utils import (get_center_of_bbox,
                   measure_distance, get_foot_position)

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        filtered_player_detections = []
        for player_dict in player_detections:
            player_detections_first_frame = player_dict
            chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if
                                    track_id in chosen_player}
            sorted_data = dict(sorted(filtered_player_dict.items(), key=lambda item: -item[1][3]))
            filtered_player_dict = {}
            i = 0
            for key, value in sorted_data.items():
                filtered_player_dict[i] = value
                i += 1
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections

    @staticmethod
    def choose_players(court_center, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_foot_position(bbox)

            # min_distance = float('inf')
            distance = measure_distance(player_center, court_center)
            # if distance < min_distance:
            #    min_distance = distance
            distances.append((track_id, distance))

        # sort the distances in ascending order
        distances.sort(key=lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = []
        # min(4, len(distances)) for doubles
        for i in range(min(2, len(distances))):
            chosen_players.append(distances[i][0])
        return chosen_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                # noinspection PyTypeChecker
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            if box is None or box.id is None:
                continue
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        return player_dict

    @staticmethod
    def draw_bboxes(video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            i = 0
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                i += 1
                cv2.putText(frame, f"Player ID: {i}", (int(bbox[0]), int(bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames
