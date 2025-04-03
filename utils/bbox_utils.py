def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    for keypoint_index in keypoint_indices:
        keypoint = keypoints[keypoint_index * 2], keypoints[keypoint_index * 2 + 1]
        distance = abs(point[1] - keypoint[1])

        if distance < closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_index

    return key_point_ind

def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1]

def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])

def expand_bbox(x1, y1, x2, y2, image_width=1280, image_height=720, scale=1.4):
    width = x2 - x1
    height = y2 - y1

    # Increase width and height by scale factor
    new_width = width * scale
    new_height = height * scale

    # Find center of the bounding box
    cx, cy = x1 + width // 2, y1 + height // 2

    # Compute new x1, y1, x2, y2
    x1_new = int(cx - new_width // 2)
    y1_new = int(cy - new_height // 2)
    x2_new = int(cx + new_width // 2)
    y2_new = int(cy + new_height // 2)

    # Ensure the new bounding box is within image bounds
    x1_new = max(0, x1_new)
    y1_new = max(0, y1_new)
    x2_new = min(image_width, x2_new)
    y2_new = min(image_height, y2_new)

    return x1_new, y1_new, x2_new, y2_new
