import numpy as np
import cv2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import argparse
#from nanosam.utils import Predictor, get_config
from shift import shiftf
import time
import random
import os
import torch

def mask_iou(mask1, mask2):
    """
    Compute the Intersection over Union (IoU) of two binary masks.
    
    Parameters:
        mask1 (numpy array): First binary mask.
        mask2 (numpy array): Second binary mask.

    Returns:
        float: IoU score.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def nms_masks(masks, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to masks.

    Parameters:
        masks (list of numpy arrays): List of binary masks.
        scores (list of floats or tensors): List of confidence scores for the masks.
        iou_threshold (float): IoU threshold for suppressing masks. Masks with IoU greater than this will be suppressed.

    Returns:
        List of masks that are kept after NMS.
    """
    # Convert scores to NumPy array if they are tensors (move to CPU and then convert to numpy)

    # Sort the masks by their score in descending order
    indices = np.argsort(scores)[::-1]
    masks_sorted = [masks[i] for i in indices]
    scores_sorted = [scores[i] for i in indices]

    keep = []
    while len(masks_sorted) > 0:
        # Keep the mask with the highest score
        mask = masks_sorted.pop(0)
        keep.append(mask)
        score = scores_sorted.pop(0)

        # Compare this mask with the rest of the masks
        i = 0
        while i < len(masks_sorted):
            current_mask = masks_sorted[i]
            # Calculate IoU between the current mask and the top mask (which is kept)
            iou = mask_iou(mask, current_mask)
            if iou > iou_threshold:
                # If IoU is above the threshold, remove the current mask
                masks_sorted.pop(i)
                scores_sorted.pop(i)
            else:
                i += 1

    return keep



def visualize_masks_on_image(image, masks, alpha=0.5):
    """
    Visualizes the masks on the image.

    Parameters:
        image (numpy array): The original image.
        masks (list of numpy arrays): List of binary masks.
        alpha (float): Transparency level for the overlay. 0.0 is fully transparent, 1.0 is fully opaque.

    Returns:
        None
    """
    # Convert image to RGB (if it's not already in RGB)
    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create a copy of the image to overlay the masks on
    overlay_image = image.copy()

    # Apply each mask to the image
    for mask in masks:
        # Ensure the mask is the same size as the image
        mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create a random color for each mask (for better visualization)
        color = np.random.randint(0, 255, size=(3,), dtype=int)

        # Use the mask to apply color to the overlay image
        overlay_image[mask_resized == 1] = (overlay_image[mask_resized == 1] * (1 - alpha) + color * alpha).astype(np.uint8)

    # Plot the original image and the overlay with the masks
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay_image)
    plt.axis('off')  # Hide the axis for a cleaner look
    plt.show()




def allmask(image, predictor):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
    cv2_image = np.array(image)

    # Convert RGB to BGR (since OpenCV uses BGR format)
    image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]

    step_height = height // 16
    step_width = width // 16
    
    # Define a margin to avoid the edges
    margin = 5  # Number of pixels to leave on the border (you can adjust this value)

    # Generate row indices and column indices with a margin
    y_indices = np.arange(step_height, height - step_height, step_height)  # Avoid the top and bottom edges
    x_indices = np.arange(step_width, width - step_width, step_width)  # Avoid the left and right edges
    grid_points = np.array(np.meshgrid(x_indices, y_indices)).T.reshape(-1, 2)
    # Create a meshgrid of (x, y) coordinates
    masks = []
    ious = []
    for i, j in enumerate(grid_points):
        points = np.array([j])
        label = np.array([1])
        mask, iou, _ = predictor.predict(points, label, multimask_output=False)
        mask =  (mask[0, 0] > 0)
        masks.append(mask)
        #print(iou[0])
     
        ious.append(iou[0])
        #print(iou[0][0])
    #print(ious)
    
    final_masks = nms_masks(masks, ious, 0.5)
    print(len(masks), len(final_masks))
    #visualize_masks_on_image(image, masks)
    visualize_masks_on_image(image, final_masks)

        
        #
        #labels = [1] * grid_points.shape[0]
        #labels = np.asarray(labels)
        #masks, iou, _ = predictor.predict(grid_points, labels)

    # Print the grid points and their shape

    """
    print(grid_points, grid_points.shape, labels, labels.shape, masks.shape,iou)
    
    # Visualize the points
    plt.imshow(image)  # Show the image in the background
    plt.scatter(grid_points[:, 0], grid_points[:, 1], color='red', s=10)  # Points in red
    plt.title("Grid Points Inside Image")
    plt.show()

    """



