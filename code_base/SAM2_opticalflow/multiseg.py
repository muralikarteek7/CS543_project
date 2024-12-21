import cv2
import numpy as np
import cv2

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import argparse
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from shift import shiftf
import time
import random
import os
from automask import allmask
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image
import random
import numpy as np
from ultralytics import YOLO
import test_yolo
import time



if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )





np.random.seed(3)
def generate_color(index):
    np.random.seed(index)  # Ensures the same color for the same index
    return list(np.random.choice(range(256), size=3))


'''
def multiflow(masks1, masks2, frame1, frame2):
    # Convert frames to grayscale once at the beginning
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Iterate through all the masks in masks1 and masks2
    for i, (mask1, mask2) in enumerate(zip(masks1, masks2)):
        # Convert masks to binary (0 or 255) if not already in that format
        if len(mask1.shape) == 3:
            mask1 = np.squeeze(mask1)  # Remove the extra dimension
        if len(mask2.shape) == 3:
            mask2 = np.squeeze(mask2)
        mask1 = np.uint8(mask1 > 0) * 255
        mask2 = np.uint8(mask2 > 0) * 255

        # Check if the mask has any non-zero pixels
        if np.sum(mask1) == 0 or np.sum(mask2) == 0:

            print(f"Mask {i} is empty, skipping.")
            continue

        # Apply the masks to the grayscale frames
        filter1 = cv2.bitwise_and(gray_frame1, gray_frame1, mask=mask1)
        filter2 = cv2.bitwise_and(gray_frame2, gray_frame2, mask=mask2)

        # Find keypoints in both masks using goodFeaturesToTrack
        keypoints1 = cv2.goodFeaturesToTrack(filter1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        keypoints2 = cv2.goodFeaturesToTrack(filter2, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        # Ensure keypoints are valid
        if keypoints1 is None or keypoints2 is None:
            print(f"No keypoints found in one or both masks for object {i}.")
            continue

        # Calculate optical flow for those keypoints between the two frames
        flow, status, error = cv2.calcOpticalFlowPyrLK(filter1, filter2, keypoints1, None)

        # Ensure flow and status are valid
        if flow is None or status is None:
            print(f"Flow or status is None for object {i}.")
            continue

        # Filter good keypoints (status == 1 means valid flow)
        good_new = flow[status == 1]
        good_old = keypoints1[status == 1]

        # Generate a unique color for each mask (using HSV to ensure different colors)
        color = tuple(np.random.choice(range(256), size=3))  # Random RGB color for each mask

        # Draw the flow vectors on the second frame for this object
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            # Draw a line from the old position to the new position
            frame2 = cv2.line(frame2, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            # Draw a circle at the new position
            frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (0, 0, 255), -1)

    # Display the resulting image with optical flow vectors for all objects
    cv2.imshow('Optical Flow - Keypoint Displacement', frame2)
    #time.sleep(0.1)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
'''




def flow(mask1,mask2, frame1,frame2):
    
    
    if len(mask1.shape) == 3:
        mask1 = np.squeeze(mask1)  # Remove the extra dimension
    if len(mask2.shape) == 3:
        mask2 = np.squeeze(mask2)
    mask1 = np.uint8(mask1 > 0) * 255  # Convert bool mask to uint8
    mask2 = np.uint8(mask2 > 0) * 255
    # print(mask1.shape)
    # Find keypoints in both masks (e.g., using goodFeaturesToTrack)
    keypoints1 = cv2.goodFeaturesToTrack(mask1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    keypoints2 = cv2.goodFeaturesToTrack(mask2, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Calculate optical flow for those keypoints
    # keypoints1 are the points in mask1, keypoints2 are the new positions in mask2
    flow, status, error = cv2.calcOpticalFlowPyrLK(mask1, mask2, keypoints1, None)

    # Filter good keypoints
    good_new = flow[status == 1]
    good_old = keypoints1[status == 1]


    # Draw the flow vectors on the second frame
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame2 = cv2.line(frame2, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame2 = cv2.circle(frame2, (int(a), int(b)), 5, (0, 0, 255), -1)

    cv2.imshow('Optical Flow - Keypoint Displacement', frame2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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




def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)




def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    #ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0
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


"""
sam2_checkpoint = "/home/brije/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)

image =  PIL.Image.open("./newsence/0.jpg")
image = np.array(image.convert("RGB"))
masks = mask_generator.generate(image)
#print(len(masks))



# Assuming 'masks' is a list of binary mask images (numpy arrays)
random_points_per_mask = []
filtermasks = []
print(type(masks))
for mask in masks:
    # Get the coordinates (row, column) of all non-zero points (i.e., the mask areas)
    mask_ = mask['segmentation']

    mask_display = (mask_ > 0).astype(np.uint8) * 255

    # Show the mask
    cv2.imshow("Mask", mask_display)

    # Wait for a key press
    key = cv2.waitKey(0)
    if key == ord('s'):
        print("saved")
        non_zero_coords = np.column_stack(np.where(mask_ > 0))
        non_zero_coords = non_zero_coords[:, [1, 0]] 

        # If the mask has fewer than 4 points, we can select all available points
        num_points = min(4, len(non_zero_coords))
        
        # Randomly sample 4 points (or fewer if the mask has fewer than 4 non-zero points)
        selected_points = random.sample(list(non_zero_coords), num_points)
        selected_points  = np.vstack([selected_points[0], selected_points[1],selected_points[2],selected_points[3]])

        # Append the selected points to the result list
        random_points_per_mask.append(selected_points)
        filtermasks.append(mask)
    else:
        print("skipped")


print(len(filtermasks))
print(type(filtermasks))
# Print the list of random points for each mask
#print(random_points_per_mask)

#print(masks)

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(filtermasks)
plt.axis('off')
plt.show() 

"""



sam2_checkpoint = "/home/brije/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)



video_dir = "./frames2trim" #"./newsence"#'./frames'

# Scan all the JPEG frame names in the directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
]


frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))




inference_state = predictor.init_state(video_path=video_dir)


out_obj = []
out_mask = []

image =  cv2.imread(video_dir + "/0.jpg")
model = YOLO("yolov8m-seg.pt")
random_points_per_mask , filtermasks = test_yolo.yolo_mask(image, model)
for i in range(len(filtermasks)):

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = i  # give a unique id to each object we interact with (it can be any integers)
    #print(i)
    # Let's add a positive click at (x, y) = (210, 350) to get started
    points = random_points_per_mask[i] #np.array([[393,269],[381,257],[372,275],[386,270]])#
    #print(points)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1,1,1,1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )


video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 1
plt.close("all")
mask2 = []


framemasks = []
mask_id = []
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    #lt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    masks_ = []
    m_id = []
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        #print(out_obj_id)
        #mask2.append(out_mask)
        masks_.append(out_mask)
        m_id.append(out_obj_id)
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    framemasks.append(masks_)
    mask_id.append(m_id)

    #plt.axis('off')  # Hide axes for better view
    #plt.show()

# ids_ = len(mask_id[0])
# random_colors = {id: np.random.rand(3,) for id in ids_}  # Random RGB colors for each ID

# print(random_colors)
# index =  list(range(100))
# for i in range(len(framemasks)-1):
#     if framemasks[i] is not None:
#         mask21 = framemasks[i]
#         mask22 = framemasks[i+1]
#         frame1 = cv2.imread(video_dir + '/'+str(index[i])+'.jpg')
#         frame2 = cv2.imread(video_dir + '/'+str(index[i+1])+'.jpg')
        
#         multiflow(mask21,mask22,frame1,frame2)



index =  list(range(199))
for i in range(len(framemasks) - 1):
    print(f'-------------{len(framemasks)}')

    # Load frames
    frame1 = cv2.imread(video_dir + '/' + str(index[i]) + '.jpg')
    frame2 = cv2.imread(video_dir + '/' + str(index[i + 1]) + '.jpg')

    if framemasks[i] is None or len(framemasks[i]) == 0:
        print(f"No masks found for frame {i}. Running YOLO inference...")
        
        random_points_per_mask, filtermasks = test_yolo.yolo_mask(frame1, model)
        
        if not filtermasks or len(filtermasks) == 0:
            print(f"YOLO inference did not detect any objects in frame {i}. Skipping...")
            continue
        
        framemasks[i] = filtermasks

    if framemasks[i + 1] is None or len(framemasks[i + 1]) == 0:
        print(f"No masks found for frame {i+1}. Running YOLO inference...")
        
        random_points_per_mask, filtermasks = test_yolo.yolo_mask(frame2, model)
        
        if not filtermasks or len(filtermasks) == 0:
            print(f"YOLO inference did not detect any objects in frame {i+1}. Skipping...")
            continue
        
        framemasks[i + 1] = filtermasks

    # Proceed with optical flow computation
    mask21 = framemasks[i]
    mask22 = framemasks[i + 1]
    multiflow(mask21, mask22, frame1, frame2)


