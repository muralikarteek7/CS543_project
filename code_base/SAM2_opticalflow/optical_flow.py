import cv2
import numpy as np
import os
import torch
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image
import matplotlib.pyplot as plt
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def initialize_sam2(model_cfg, checkpoint_path, device):
    predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
    return predictor

# Displays the segmentation results on the image.
def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


def visualize_segmentation(frame, mask_logits, alpha=0.5):
    """
    Overlays the segmentation masks on the frame image and displays the result.
    
    Args:
        frame (numpy.ndarray): The original image (RGB format) as a NumPy array.
        mask_logits (torch.Tensor): The segmentation mask logits as a PyTorch tensor.
        alpha (float): The transparency level for overlaying the mask on the frame.
    """
    # Convert the mask logits to binary masks
    masks = (mask_logits.sigmoid() > 0.5).cpu().numpy()  # Apply threshold for mask

    # Overlay each mask on the image
    overlay_frame = frame.copy()
    for i, mask in enumerate(masks):
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)  # Random mask color
        for c in range(3):  # Overlay mask on each channel
            overlay_frame[:, :, c] = np.where(mask, color[c], overlay_frame[:, :, c])
    
    # Combine original frame and overlay with transparency
    combined_image = cv2.addWeighted(frame, 1 - alpha, overlay_frame, alpha, 0)
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_image)
    plt.axis("off")
    plt.title("Segmentation Overlay")
    plt.show()



def segment_first_frame(predictor, frame, video_dir, obj_id=1):
    frame_np = np.array(frame.convert("RGB"))
    frame_tensor = torch.tensor(frame_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # mask_generator = SAM2AutomaticMaskGenerator(predictor)
    # masks = mask_generator.generate(frame_np)

    # plt.figure(figsize=(20, 20))
    # plt.imshow(frame_np)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show() 
    print(f'------------   -1')
    points = np.array([[384, 262], [224, 257]], dtype=np.float32)  
    labels = np.array([1, 1], np.int32) 
    print(f'------------0')
    inference_state = predictor.init_state(video_path=video_dir) 
    _, obj_ids, mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0, 
        obj_id=obj_id,
        points=points,
        labels=labels,
    )
    print(f'------------1')
    visualize_segmentation(frame_np, mask_logits)
    print(f'------------2')
    return mask_logits[0].cpu().numpy(), inference_state, obj_ids



# ==== Optical Flow Tracking ====
def track_objects_with_flow2(prev_frame, curr_frame, prev_mask):
    """
    Tracks objects from the previous frame to the current frame using optical flow.
    """
    # Ensure prev_mask is 2D
    if len(prev_mask.shape) > 2:
        prev_mask = prev_mask[..., 0]  # Use only the first channel if multi-channel

    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Warp the mask based on the flow field
    h, w = prev_mask.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # Ensure flow dimensions match grid dimensions
    if flow.shape[:2] != (h, w):
        raise ValueError(f"Flow dimensions {flow.shape[:2]} do not match mask dimensions {(h, w)}")

    flow_map = np.stack((grid_x + flow[..., 0], grid_y + flow[..., 1]), axis=-1)
    curr_mask = cv2.remap(prev_mask.astype(np.float32), flow_map, None, cv2.INTER_LINEAR)
    return (curr_mask > 0.5).astype(np.uint8)  # Threshold the warped mask


def track_objects_with_flow(prev_frame, curr_frame, prev_mask):
    """
    Tracks objects from the previous frame to the current frame using optical flow.
    """
    # Ensure prev_mask is 2D
    if len(prev_mask.shape) > 2:
        prev_mask = prev_mask[..., 0]  # Use only the first channel if multi-channel

    # Ensure prev_mask matches frame dimensions
    prev_mask = cv2.resize(prev_mask, (prev_frame.shape[1], prev_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Warp the mask based on the flow field
    h, w = prev_mask.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Create flow_map and ensure it has the correct type
    flow_map = np.stack((grid_x + flow[..., 0], grid_y + flow[..., 1]), axis=-1).astype(np.float32)

    # Apply remapping
    curr_mask = cv2.remap(prev_mask.astype(np.float32), flow_map, None, cv2.INTER_LINEAR)

    return (curr_mask > 0.5).astype(np.uint8)  # Threshold the warped mask


def process_video(video_dir, output_dir, model_cfg, checkpoint_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = initialize_sam2(model_cfg, checkpoint_path, device)

    frame_names = sorted(
        [p for p in os.listdir(video_dir) if p.lower().endswith(".jpeg")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    frames = [cv2.imread(os.path.join(video_dir, name)) for name in frame_names]
    if not frames:
        raise ValueError("No frames found in video_dir!")


    first_frame = Image.open(os.path.join(video_dir, frame_names[0]))
    print(f'-------------------{os.path.join(video_dir, frame_names[0])}')
    init_mask, inference_state, obj_ids = segment_first_frame(predictor, first_frame, video_dir)

    prev_frame = frames[0]
    prev_mask = init_mask

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, frame_names[0]), prev_mask * 255)

    for idx, curr_frame in enumerate(frames[1:], start=1):
        curr_mask = track_objects_with_flow(prev_frame, curr_frame, prev_mask)

        # Handle occlusion/drift (optional): Reapply segmentation every 10 frames
        if idx % 10 == 0:
            # curr_image = Image.fromarray(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB))
            curr_image = Image.open(os.path.join(video_dir, frame_names[idx]))

            curr_mask, _, _ = segment_first_frame(predictor, curr_image, video_dir)

        mask_path = os.path.join(output_dir, frame_names[idx])
        cv2.imwrite(mask_path, curr_mask * 255)

        prev_frame = curr_frame
        prev_mask = curr_mask

video_dir = "/home/brije/sam2/frames_jpeg"  
output_dir = "/home/brije/sam2/output_masks" 
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  
checkpoint_path = "/home/brije/sam2/checkpoints/sam2.1_hiera_large.pt"
process_video(video_dir, output_dir, model_cfg, checkpoint_path)
