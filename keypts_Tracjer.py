import os
import cv2
from constants import Directories
# Set paths
_DIR = Directories()
images_dir = f"{_DIR.INPUT_DIR}/shuttlecock_dataset/images/train"
labels_dir = f"{_DIR.INPUT_DIR}/shuttlecock_dataset/labels/train"

# Image size (needed for un-normalizing coords)
img_w, img_h = None, None
# Loop through all images
for img_name in sorted(os.listdir(images_dir)):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(images_dir, img_name)
    label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    if img is None:
        continue

    if img_w is None or img_h is None:
        img_h, img_w = img.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                if line != "":
                    class_id, x, y, w, h = map(float, line.strip().split())
                    # Unnormalize
                    x1 = int((x - w / 2) * img_w)
                    y1 = int((y - h / 2) * img_h)
                    x2 = int((x + w / 2) * img_w)
                    y2 = int((y + h / 2) * img_h)

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, "shuttlecock", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)



    # Show the image
    cv2.imshow("Labeled", img)
    key = cv2.waitKey(0)  # press any key to move to next image
    if key == ord('q'):  # press 'q' to quit early
        break

cv2.destroyAllWindows()
