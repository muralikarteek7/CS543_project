from ultralytics import YOLO
import random
import cv2
import numpy as np

def get_centroid_pixel_value(mask, image):

    mask_points = np.array(mask, dtype=np.int32)
    M = cv2.moments(mask_points)
    if M["m00"] == 0:
        return None, None
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    pixel_value = tuple(image[cY, cX]) 
    
    return (cX, cY), pixel_value



def getRandPoints(mask_):
    print(mask_[0])
    
    non_zero_coords = mask_[0]
    non_zero_coords = non_zero_coords.tolist()
    #non_zero_coords = sorted(non_zero_coords[:, [1, 0]]) 
    

    # Randomly select 4 coordinates from the flipped list
    selected_points = random.sample(non_zero_coords, 4)
    selected_points  = np.vstack([selected_points[0], selected_points[1],selected_points[2],selected_points[3]])



    print(selected_points)
    return selected_points



def yolo_mask(img, model):
    yolo_classes = list(model.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
    conf = 0.5
    target_class = "car"
    car_id = yolo_classes.index(target_class)
    results = model.predict(img, conf=conf)
    colors = [random.choices(range(256), k=3) for _ in classes_ids]
    random_points_per_mask = []
    maskslist = []
    for result in results:
        for mask, box in zip(result.masks.xy, result.boxes):
            if int(box.cls[0]) == car_id:
                points = np.int32([mask])
                color_number = classes_ids.index(int(box.cls[0]))
                cv2.fillPoly(img, points, colors[color_number])
                #print(getRandPoints(mask))

                random_points_per_mask.append(getRandPoints(points))
                maskslist.append(mask)
            

            """
            centroid, pixel_value = get_centroid_pixel_value(mask, img)
            if centroid:
                print(f"Centroid: {centroid}")
                cv2.circle(img, centroid, 5, (0, 0, 255), -1)  
            """
    return random_points_per_mask, maskslist
    # Display and save the output image


    """
    cv2.imshow("Segmented Image with Centroids", img)
    cv2.waitKey(0)

    # Save the processed image
    output_path = "/home/brije/sam2/segmented_output.jpg"
    cv2.imwrite(output_path, img)
    print(f"Image saved to {output_path}")

    cv2.destroyAllWindows()

    """



"""
model = YOLO("yolov8m-seg.pt")
img = cv2.imread("/home/brije/sam2/frames/0.jpg")
yolo_mask(img,model)
"""









'''
from ultralytics import YOLO
import random
import cv2
import numpy as np

model = YOLO("yolov8m-seg.pt")
img = cv2.imread("/home/brije/sam2/newsence/0.jpg")

# if you want all classes
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

conf = 0.5

results = model.predict(img, conf=conf)
colors = [random.choices(range(256), k=3) for _ in classes_ids]
print(results)
for result in results:
    for mask, box in zip(result.masks.xy, result.boxes):
        points = np.int32([mask])
        # cv2.polylines(img, points, True, (255, 0, 0), 1)
        color_number = classes_ids.index(int(box.cls[0]))
        cv2.fillPoly(img, points, colors[color_number])

cv2.imshow("Image", img)
cv2.waitKey(0)

cv2.imwrite("/home/brije/sam2/", img)
'''