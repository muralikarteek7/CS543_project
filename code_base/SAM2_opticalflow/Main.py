import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
from test_yolo import yolo_mask
from sam2.build_sam import build_sam2_video_predictor


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS support is preliminary and may result in degraded performance.")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def initialize_predictor(device, checkpoint, config):
    return build_sam2_video_predictor(config, checkpoint, device=device)


def generate_color(index):
    np.random.seed(index)
    return list(np.random.choice(range(256), size=3))


def multiflow(masks1, masks2, frame1, frame2):
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    overlay_frame1 = frame1.copy()
    overlay_frame2 = frame2.copy()

    for i, (mask1, mask2) in enumerate(zip(masks1, masks2)):
        if len(mask1.shape) == 3:
            mask1 = np.squeeze(mask1)
        if len(mask2.shape) == 3:
            mask2 = np.squeeze(mask2)
        mask1 = np.uint8(mask1 > 0) * 255
        mask2 = np.uint8(mask2 > 0) * 255

        if np.sum(mask1) == 0 or np.sum(mask2) == 0:
            print(f"Mask {i} is empty, skipping.")
            continue

        color = generate_color(i)

        for c in range(3):  
            overlay_frame1[:, :, c] = np.where(mask1 > 0, color[c], overlay_frame1[:, :, c])

        for c in range(3):
            overlay_frame2[:, :, c] = np.where(mask2 > 0, color[c], overlay_frame2[:, :, c])

        keypoints1 = cv2.goodFeaturesToTrack(cv2.bitwise_and(gray_frame1, gray_frame1, mask=mask1),
                                             maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        keypoints2 = cv2.goodFeaturesToTrack(cv2.bitwise_and(gray_frame2, gray_frame2, mask=mask2),
                                             maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        if keypoints1 is None or keypoints2 is None:
            print(f"No keypoints found in one or both masks for object {i}.")
            continue

        flow, status, error = cv2.calcOpticalFlowPyrLK(cv2.bitwise_and(gray_frame1, gray_frame1, mask=mask1),
                                                       cv2.bitwise_and(gray_frame2, gray_frame2, mask=mask2),
                                                       keypoints1, None)

        if flow is not None and status is not None:
            good_new = flow[status == 1]
            good_old = keypoints1[status == 1]

            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                frame2 = cv2.line(frame2, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (0, 0, 255), -1)

    alpha = 0.35
    combined_frame2 = cv2.addWeighted(frame2, 1 - alpha, overlay_frame2, alpha, 0)
    cv2.imshow('Keypoint Displacement + Tracking mask', combined_frame2)
    cv2.waitKey(200)
    cv2.destroyAllWindows()

    return combined_frame2


def process_video_frames(video_dir, predictor, model):
    frame_names = sorted(
        [p for p in os.listdir(video_dir) if p.lower().endswith(('.jpg', '.jpeg'))],
        key=lambda p: int(os.path.splitext(p)[0])
    )

    inference_state = predictor.init_state(video_path=video_dir)

    # Initialize input points/masks
    initial_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
    random_points_per_mask, filtermasks = yolo_mask(initial_frame, model)
    for i, mask in enumerate(filtermasks):
        points = random_points_per_mask[i]
        labels = np.ones(len(points), dtype=np.int32)  # Positive points
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=i,
            points=points,
            labels=labels,
        )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    framemasks = []
    for out_frame_idx in range(len(frame_names)):
        masks_ = [video_segments[out_frame_idx].get(obj_id) for obj_id in video_segments[out_frame_idx]]
        framemasks.append(masks_)

    index = list(range(len(frame_names)))
    for i in range(len(framemasks) - 1):
        frame1 = cv2.imread(f"{video_dir}/{index[i]}.jpg")
        frame2 = cv2.imread(f"{video_dir}/{index[i + 1]}.jpg")

        if not framemasks[i]:
            print(f"No masks found for frame {i}. Running YOLO inference...")
            _, framemasks[i] = yolo_mask(frame1, model)

        if not framemasks[i + 1]:
            print(f"No masks found for frame {i + 1}. Running YOLO inference...")
            _, framemasks[i + 1] = yolo_mask(frame2, model)

        out = multiflow(framemasks[i], framemasks[i + 1], frame1, frame2)
        cv2.imwrite(f"results/{index[i]}.jpg", out)



if __name__ == "__main__":
    device = setup_device()
    sam2_checkpoint = "/home/brije/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = initialize_predictor(device, sam2_checkpoint, model_cfg)

    video_directory = "./frames2trim"
    yolo_model = YOLO("yolov8m-seg.pt")

    process_video_frames(video_directory, predictor, yolo_model)
