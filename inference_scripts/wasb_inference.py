import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from model_definations.wasb import HRNet

def preprocess_frame(frame, transform):
    return transform(frame)

def predict_ball_position(prev_positions, width, height):
    if len(prev_positions) < 3:
        return None
    p_t = prev_positions[-1]
    a_t = p_t - 2 * prev_positions[-2] + prev_positions[-3]
    v_t = p_t - prev_positions[-2] + a_t
    predicted_position = p_t + v_t + 0.5 * a_t
    predicted_position = np.clip(predicted_position, [0, 0], [width, height])
    return predicted_position

def run_inference(weights, input_path, confidence_threshold=30.):
    config = {
        "name": "hrnet",
        "frames_in": 3,
        "frames_out": 3,
        "inp_height": 288,
        "inp_width": 512,
        "out_height": 288,
        "out_width": 512,
        "rgb_diff": False,
        "out_scales": [0],
        "MODEL": {
            "EXTRA": {
                "FINAL_CONV_KERNEL": 1,
                "PRETRAINED_LAYERS": ['*'],
                "STEM": {
                    "INPLANES": 64,
                    "STRIDES": [1, 1]
                },
                "STAGE1": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 1,
                    "BLOCK": 'BOTTLENECK',
                    "NUM_BLOCKS": [1],
                    "NUM_CHANNELS": [32],
                    "FUSE_METHOD": 'SUM'
                },
                "STAGE2": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 2,
                    "BLOCK": 'BASIC',
                    "NUM_BLOCKS": [2, 2],
                    "NUM_CHANNELS": [16, 32],
                    "FUSE_METHOD": 'SUM'
                },
                "STAGE3": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 3,
                    "BLOCK": 'BASIC',
                    "NUM_BLOCKS": [2, 2, 2],
                    "NUM_CHANNELS": [16, 32, 64],
                    "FUSE_METHOD": 'SUM'
                },
                "STAGE4": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 4,
                    "BLOCK": 'BASIC',
                    "NUM_BLOCKS": [2, 2, 2, 2],
                    "NUM_CHANNELS": [16, 32, 64, 128],
                    "FUSE_METHOD": 'SUM'
                },
                "DECONV": {
                    "NUM_DECONVS": 0,
                    "KERNEL_SIZE": [],
                    "NUM_BASIC_BLOCKS": 2
                }
            },
            "INIT_WEIGHTS": True
        },
        "model_path": f"{weights}",  # Update with your model path
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(device)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['inp_height'], config['inp_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = HRNet(cfg=config).to(device)
    checkpoint = torch.load(config['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    coordinates = []
    frame_number = 0
    frames_buffer = []
    prev_positions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_buffer.append(frame)
        if len(frames_buffer) == config['frames_in']:
            # Preprocess the frames
            frames_processed = [preprocess_frame(f, transform) for f in frames_buffer]
            input_tensor = torch.cat(frames_processed, dim=0).unsqueeze(0).to(device)

            # Perform inference
            with torch.no_grad():
                outputs = model(input_tensor)[0]  # Get the raw logits

            detected = False
            center_x, center_y, confidence = 0, 0, 0

            for i in range(config['frames_out']):
                output = outputs[0][i]
                # Post-process the output
                output = torch.sigmoid(output)  # Apply sigmoid to the output to get probabilities
                heatmap = output.squeeze().cpu().numpy()
                heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
                heatmap = (heatmap > 0.5).astype(np.float32) * heatmap

                # Find connected components
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats((heatmap > 0).astype(np.uint8), connectivity=8)

                # Calculate centers of blobs
                blob_centers = []
                for j in range(1, num_labels):  # Skip the background label 0
                    mask = labels_im == j
                    blob_sum = heatmap[mask].sum()
                    if blob_sum > 0:
                        center_x = np.sum(np.where(mask)[1] * heatmap[mask]) / blob_sum
                        center_y = np.sum(np.where(mask)[0] * heatmap[mask]) / blob_sum
                        blob_centers.append((center_x, center_y, blob_sum))

                if blob_centers:
                    predicted_position = predict_ball_position(prev_positions, width, height)
                    if predicted_position is not None:
                        # Select the blob closest to the predicted position
                        distances = [np.sqrt((x - predicted_position[0]) ** 2 + (y - predicted_position[1]) ** 2) for x, y, _ in blob_centers]
                        closest_blob_idx = np.argmin(distances)
                        center_x, center_y, confidence = blob_centers[closest_blob_idx]
                    else:
                        # Select the blob with the highest confidence if no prediction is available
                        blob_centers.sort(key=lambda x: x[2], reverse=True)
                        center_x, center_y, confidence = blob_centers[0]
                    detected = True
                    prev_positions.append(np.array([center_x, center_y]))
                    if len(prev_positions) > 3:
                        prev_positions.pop(0)

                # Write the frame to the output video and save the coordinates
                if detected:
                    coordinates.append([frame_number, 1, center_x, center_y, confidence])
                else:
                    coordinates.append([frame_number, 0, center_x, center_y, confidence])

                frame_number += 1
            frames_buffer = []  # Clear the buffer for the next set of frames

    # Release everything if job is finished
    cap.release()
    return coordinates


