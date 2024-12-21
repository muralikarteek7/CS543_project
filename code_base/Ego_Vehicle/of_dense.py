import numpy as np
import cv2
import time
import os
import skimage
from cv2 import cuda

def draw_opticalflow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr

def draw_hsv(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr
########################################


def get_dense_of(prevgray,image_files):
        
    for image_file in image_files[1:]:
        img = cv2.imread(image_file)
        downsampled_image = cv2.resize(img, (int(0.5*(img.shape[1])),int(0.5*(img.shape[0]))), interpolation=cv2.INTER_CUBIC)
        imggray = cv2.cvtColor(downsampled_image, cv2.COLOR_BGR2GRAY)
        t1 = time.time()
        
        denseflow=cv2.calcOpticalFlowFarneback(prevgray, imggray, None, 0.5, 3, 15, 5, 5, 1.2, flags=1)
        print(denseflow.shape)
        prevgray=imggray
        # print(denseflow.shapse, type(denseflow))
        fps=1/(time.time()-t1)
        # print("FPS={}".format(fps))

        cv2.imshow('flow HSV',draw_hsv(denseflow))
        # cv2.imshow('flow',draw_opticalflow(imggray,denseflow))
        
        if cv2.waitKey(500)==ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    folder_path = 'G:\AV projects\CV_project\cam_front'
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])

    print(cv2.cuda.getCudaEnabledDeviceCount())

    prev = cv2.imread(image_files[0])
    downsampled_prev = cv2.resize(prev,(int(0.5*(prev.shape[1])),int(0.5*(prev.shape[0]))), interpolation=cv2.INTER_CUBIC)
    prevgray = cv2.cvtColor(downsampled_prev, cv2.COLOR_BGR2GRAY)

    get_dense_of(prevgray,image_files)

if __name__ == "__main__":
    main()

############# VIDEO INPUT #############################
# while True:
#     ret, img=cap.read()
#     if not ret:
#         break
    
#     imggray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     print(imggray.shape)
#     t1 = time.time()

#     # #TODO: make this params dynamic
#     # denseflow = cv2.calcOpticalFlowFarneback(prevgray, imggray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     # prevgray = imggray

#     # fps = 1 / (time.time()-t1)
#     # print("FPS={}".format(round(fps,2)))

#     # # cv2.imshow('flow', draw_opticalflow(imggray, denseflow))
#     # # cv2.imshow('flow HSV', draw_hsv(denseflow))

#     if cv2.waitKey(5) == ord('q'):
#         break


# cap.release()
# cv2.destroyAllWindows()
