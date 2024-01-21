import numpy as np 
import cv2
import torch
import random

from yolo_pose_onnx import *
from sort4 import Sort, KalmanBoxTracker
model_path = 'D:\Phong\TT\sort\yolov7-w6-pose.onnx'

mot_tracker = Sort()
video_path = 'D:\Phong\Mon_hoc\Project3\\video_supermarket.mp4'
video = cv2.VideoCapture(video_path)
#folder_path = 'D:\Phong\Mon_hoc\Project3\cafe_shop_0_camera1'
count = 0
ids = []
file_save = 'D:\Phong\Mon_hoc\Project3\\tracking.txt'
f = open(file_save, 'w')
#for img in sorted(os.listdir(folder_path)):
while (video.isOpened()):
    #img_path = os.path.join(folder_path, img)
    #img_show = cv2.imread(img_path)
    ret, img_show = video.read()
    if img_show is None:
        break
    img_show = cv2.resize(img_show, (640, 640))
    preds = read_model(img_show)
    height, width, _ = img_show.shape
    preds[:, 0], preds[:, 2] = preds[:, 0] / 640.0 * width, preds[:,2] / 640.0 * width
    preds[:, 1], preds[:, 3] = preds[:,1] / 640.0 * height, preds[:,3] / 640.0 * height
    for i in range(6, 57, 3):
        if i%3 != 2:
            preds[:, i] = preds[:, i] / 640.0 * width
            preds[:, i+1] = preds[:, i+1] / 640.0 * height
    new_preds = []
    for row in preds:
        if row[4] > 0.6:
            print(row[4])
            new_preds.append(row)
    new_preds = np.array(new_preds)
    try:
        bboxs, confs, labels, kpts = new_preds[:, 0:5], new_preds[:, 4], new_preds[:, 5], new_preds[:, 6:]
    except:
        count += 1
        continue
    for i in range(6, 57, 3):
        x = new_preds[:, i]
        y = new_preds[:, i+1]
        xx1, yy1, xx2, yy2 = new_preds[:, 0], new_preds[:, 1], new_preds[:, 2], new_preds[:, 3]
        new_preds[:, i] = (x-xx1)/(xx2-xx1)
        new_preds[:, i+1] = (y-yy1)/(yy2-yy1)
    track_bbs_ids = mot_tracker.update(new_preds)
    num_people = len(track_bbs_ids.tolist())
    coors_later, confs_later, coors_prev, confs_prev = [[]]*num_people, [[]]*num_people, [[]]*num_people, [[]]*num_people
    for j in range(num_people):
        coords = track_bbs_ids.tolist()[j]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        #print(coords[4])
        name_idx = int(coords[4])-1
        name = 'ID: {}'.format(str(name_idx))
        ids.append(name_idx)
        color1 = (255, 0, 0)
        color2 = (0, 0, 255)
        random.seed(int(coords[4]))
        color = (255*random.random(), 255*random.random(), 255*random.random())
        width = x2 - x1
        height = y2 - y1
        f.write(str(count) + ', ' +str(name_idx) +', '+ str(x1) + ', '+ str(y1) + ', ' + str(width) + ', '+ str(height)+ ', 1, 1, 1')
        f.write('\n')
        kpt = kpts[j]
        cv2.rectangle(img_show, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img_show, name, (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    kpts_prev = kpts
    bboxs_prev = bboxs
    count += 1
    print(count)
print(max(ids))