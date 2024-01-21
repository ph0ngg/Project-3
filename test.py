from yolo_pose_onnx import *
model_path = 'D:\Phong\TT\sort\yolov7-w6-pose.onnx'
import random
def normalization(x, y, bbox):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    width = x2 - x1
    height = y2 - y1
    x_new = (x-x1)/width
    y_new = (y-y1)/height
    return x_new, y_new

def cal_pose_simi(kpt1, kpt2, bbox1, bbox2):
    coors1, coors2, confs1, confs2 = [], [], [], []
    for i in range(len(kpt1)):
        if i%3 == 2:
            confs1.append(kpt1[i])
            confs2.append(kpt2[i])
        else:
            coors1.append(kpt1[i])
            coors2.append(kpt2[i])
    summation1 = 1.0/np.sum(confs1)
    summation2 = 0
    for i in range(0, len(coors1), 2):
        x1, y1 = normalization(coors1[i], coors1[i+1], bbox1)
        x2, y2 = normalization(coors2[i], coors2[i+1], bbox2)
        # summation2 += np.abs(x1 - x2 + y1 - y2)* confs1[i//2]
        summation2 += (np.abs(x1-x2)+ np.abs(y1-y2))* confs1[i//2]
    summation = summation1 * summation2
    return summation

from sort4 import Sort, KalmanBoxTracker
#from test2 import Sort, KalmanBoxTracker
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
#save_path1 = 'D:\Phong\TT\sort\data\\frame4_(video.mp4)'
#save_path1 = '/mnt/hdd3tb/Users/phongnn/test/sort/data/frame14'
#video = cv2.VideoCapture('D:\Phong\TT\sort\data\MOT17-02-DPM-raw.webm')

mot_tracker = Sort()
i = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#fps = int(video.get(cv2.CAP_PROP_FPS))
#print(fps)
#frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
frame_size = (640, 640)
#create a new video 
#out = cv2.VideoWriter('dancetrack0015_n.mp4', (fourcc), 20, frame_size)
count = 1
detected = 0
detected_true = 0
kpts_prev = np.empty((0,51))

folder_path = 'D:\Phong\TT\dancetrack\\train\\train'
for filename in sorted(os.listdir(folder_path), reverse= False): 
 # if filename == 'dancetrack0015':
    print(filename)
    KalmanBoxTracker.count = 0
    file_path = os.path.join(folder_path, filename + '/img1')
    #filesave = filename +'.txt'
    #save_path = os.path.join('D:\Phong\TT\sort\data', filesave)
    #f = open(save_path, 'w')
    for img in sorted(os.listdir(file_path)):
#while(video.isOpened()):
#        ret, img_show = video.read()
        img_path = os.path.join(file_path, img)
        img_show = cv2.imread(img_path)
        #print(img_show.shape)
        if img_show is None:
            break
        img_show = cv2.resize(img_show, (640, 640))
        preds = read_model(img_show)
        img_height, img_width, _ = img_show.shape
    #   bboxs, confs, labels, kpts = preds[:, 0:5], preds[:, 4], preds[:, 5], preds[:, 6]
        preds[:, 0], preds[:, 2] = preds[:, 0] / 640.0 * img_width, preds[:,2] / 640.0 * img_width
        preds[:, 1], preds[:, 3] = preds[:,1] / 640.0 * img_height, preds[:,3] / 640.0 * img_height
        for i in range(6, 57, 3):
            if i % 3 != 2:
                preds[:, i] = preds[:, i] / 640.0 * img_width
                preds[:, i+1] = preds[:, i+1] / 640.0 * img_height
        new_preds = []
        for rows in preds:
        #  print('conf = ',rows[4])
            if rows[4] > 0.2:
                new_preds.append(rows)
        new_preds = np.array(new_preds)
        try:
            bboxs, confs, labels, kpts = new_preds[:, 0:5], new_preds[:, 4], new_preds[:, 5], new_preds[:, 6:]
    #    print(preds.shape, new_preds.shape)
        except:
#            img_path1 = os.path.join(save_path1, str(count)+ '.jpg')
 #           cv2.imwrite(img_path1, img_show)
            #out.write(img_show)       
            count += 1
            continue
        for i in range(6, 57, 3):
            x = new_preds[:, i]
            y = new_preds[:, i+1]
            xx1, yy1, xx2, yy2 = new_preds[:, 0], new_preds[:, 1], new_preds[:, 2], new_preds[:, 3]
            new_preds[:, i] = (x-xx1)/(xx2-xx1)
            new_preds[:, i+1] = (y-yy1)/(yy2-yy1)
        track_bbs_ids = mot_tracker.update(new_preds)
        #print(track_bbs_ids)
        num_people = len(track_bbs_ids.tolist()) 
        coors_later, confs_later, coors_prev, confs_prev = [[]]*num_people, [[]]*num_people, [[]]*num_people, [[]]*num_people
        for j in range(len(track_bbs_ids.tolist())):
            #print(confs[j])
            #if confs[j] > 0.2:
                coords = track_bbs_ids.tolist()[j]
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                #print(coords[4])
                name_idx = int(coords[4])-1
                name = 'ID: {}'.format(str(name_idx))
                color1 = (255, 0, 0)
                color2 = (0, 0, 255)
                random.seed(int(coords[4]))
                color = (255*random.random(), 255*random.random(), 255*random.random())
                width = x2 - x1
                height = y2 - y1
                #f.write(str(count) + ', ' +str(name_idx) +', '+ str(x1) + ', '+ str(y1) + ', ' + str(width) + ', '+ str(height)+ ', 1, 1, 1')
                #f.write('\n')
        #       ----------------------------------------------------------------
                kpt = kpts[j]

                # pose_simi = 0
                # if count >= 2:
                # # try:
                #     kpt_prev = kpts_prev[j]
    #                pose_simi = round(cal_pose_simi(kpt, kpt_prev, bboxs[j], bboxs[j])*100, 2)
    #                pose_matrix[j][j] = pose_simi
                    #print(pose_simi)
                # except:
                #     pass
        #       ----------------------------------------------------------------
                #check = True
        
                cv2.rectangle(img_show, (x1, y1), (x2, y2), color, 1)
                cv2.putText(img_show, name, (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                #cv2.putText(img_show, str(confs[j]), (x2, y2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                #plot_skeleton_kpts(img_show, kpt)        
    #            cv2.putText(img_show, str(int(track_bbs_ids.tolist()[j][4]))+'  :'+str(pose_simi), (x2-40, y2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1)
                # for i in range(len(track_bbs_ids.tolist())):
                #     if i != j & count >= 2:
                #         pose_simi_ji = round(cal_pose_simi(kpts[j], kpts_prev[i], bboxs[j], bboxs[i])*100, 2)
                #         pose_matrix[j][i] = pose_simi[j][i]
                #         cv2.putText(img_show, str(int(track_bbs_ids.tolist()[i][4]))+'  :'+str(pose_simi_ji) , (x2-40, y2-30-i*10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1)  
                #         if pose_simi_ji < pose_simi:
                #             check = False
                # if (check):
                #     detected_true += 1
                #     cv2.rectangle(img_show, (x1, y1), (x2, y2), color1, 1)
                # else:
                #     cv2.rectangle(img_show, (x1, y1), (x2, y2), color2, 1)
                #detected += 1
        kpts_prev = kpts
        bboxs_prev = bboxs
     #   img_path1 = os.path.join(save_path1, str(count)+ '.jpg')
        #cv2.imwrite(img_path1, img_show)
        #out.write(img_show)       
        count += 1
        print(count)
    count = 1
#video.release()
#out.release()
#print(detected_true, detected)
#print('accuracy = ', detected_true/detected)
