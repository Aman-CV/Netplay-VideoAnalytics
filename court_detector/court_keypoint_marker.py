import os
import cv2
import pickle

from av.error import enum_name

window_name = "Image"
points = []

def save_points(name, p = None):
    global points
    with open(f"keycheckpoint/{name}.pkl", "wb") as f:
        if not p:
            pickle.dump(points, f)
            points = []
        else : pickle.dump(p, f)

    print(f"Saved {len(points)} points to points.pkl")

def mark_point(event, x, y, flags, params):
    frame = params
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))  # Store point
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Draw red dot
        cv2.imshow(window_name, frame)  # Update image

def main(args, func, name):
    cap = cv2.VideoCapture(args)

    # Get the total frame count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the middle frame index
    mid_frame_index = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)

    # Read the frame
    ret, frame = cap.read()
    if ret:
        # Save the middle frame as an image
        cv2.imshow(window_name, frame)
        cv2.setMouseCallback("Image", func, frame)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                save_points(name)
                print(points)
                x, y = map(lambda v: sum(v) / len(points), zip(*points))
                print("Average : " , x, y)
                break
    else:
        print("Failed to extract frame")


if __name__ == "__main__":
    with open("keycheckpoint/court_points4_5avg.pkl", 'rb') as f:
        court_points = pickle.load(f)


    # base_dir = "C:/Users/AmanGautam/PycharmProjects/BadmintonCoachAI/input_videos/Professional/"
    # for i in range(1, 24):
    #     match_ = base_dir + "match" + str(i) + "/video/"
    #     identifier = "match" + str(i)
    #     for file in os.listdir(match_):
    #         main(match_ + str(file), mark_point, identifier)
    #         break
