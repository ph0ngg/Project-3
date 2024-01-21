# # from ultralytics import YOLO

# # model = YOLO('yolov8n-pose.pt') 
# # model_onnx = model.export(format = 'onnx')

# from ultralytics import YOLO
# import cv2


# model = YOLO('yolov8n-pose.onnx')
# results = model('D:\Phong\TT\sort\img.png')
# res_plotted = results[0].plot()
# cv2.imshow('result', res_plotted)
# cv2.waitKey(0)

import os
import numpy as np
import cv2
import argparse
import onnxruntime
from tqdm import tqdm

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

def read_img(img, img_mean=127.5, img_scale=1/127.5):
    #img = cv2.imread(img_file)[:, :, ::-1]
    #img = img[:, :, ::-1]
    img = cv2.resize(img, (640,640), interpolation=cv2.INTER_LINEAR)
    img = (img - img_mean) * img_scale
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img,0)
    img = img.transpose(0,3,1,2)
    return img
#onnx_model = onnx.load(args.model_path)
model_path = 'D:\Phong\TT\sort\yolov7-w6-pose.onnx'
EP_list =  [
    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT", 'device_id': 0}),
    "CPUExecutionProvider"
  ]
session = onnxruntime.InferenceSession(model_path, providers = EP_list)
def read_model(img_file):
    #img_file = 'D:\Phong\TT\sort\img.jpg'
    input = read_img(img_file, img_mean= 0.0, img_scale= 0.00392156862745098)
    #model_path = 'D:\Phong\TT\sort\yolov7-w6-pose.onnx'
    

    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: input})
    #print(output.shape)
    output = output[0]
    return output

def plot_skeleton_kpts(im, kpts, steps=3):
    num_kpts = len(kpts) // steps
    #plot keypoints
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        conf = kpts[steps * kid + 2]
        if conf > 0.5: #Confidence of a keypoint has to be greater than 0.5
            cv2.circle(im, (int(x_coord), int(y_coord)), 2, (int(r), int(g), int(b)), -1)
            cv2.putText(im,str(kid), (int(x_coord), int(y_coord)), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color= (255, 0, 0),thickness=1)
    #plot skeleton
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        conf1 = kpts[(sk[0]-1)*steps+2]
        conf2 = kpts[(sk[1]-1)*steps+2]
        if conf1>0.5 and conf2>0.5: # For a limb, both the keypoint confidence must be greater than 0.5
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=1)

def post_process(img, output):
    bboxs, scores, labels, kpts = output[:, :4], output[:, 4], output[:, 5], output[:, 6:] 
#    img = cv2.imread(img_file)
    #print(img.shape)
    # height, width, _ = img.shape
    # bboxs[:, 0], bboxs[:, 2] = bboxs[:, 0]/640*width, bboxs[:, 2]/640*width
    # bboxs[:, 1], bboxs[:, 3] = bboxs[:, 1]/640*height, bboxs[:, 3]/640*height
    img = cv2.resize(img, (640, 640))
    # print(output[:, :4])
    for idx in range(len(bboxs)):
        bbox, score, label, kpt = bboxs[idx], scores[idx], labels[idx], kpts[idx]
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color = (255, 255, 0), thickness= 1)
        cv2.putText(img, str(score), (int(bbox[0]), int(bbox[1]) + 10), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color= (255, 0, 0),thickness=1)
        plot_skeleton_kpts(img, kpt)
        cv2.imwrite('out.jpg', img)

def get_kpts_info(output):
    kpts = output[:, 6:]
    coors = []
    confs = []
    for i in range(len(kpts[0])):
        if i%3 == 2:
            confs.append(list(kpts[:, i]))
        else:
            coors.append(list(kpts[:, i]))
    return coors, confs
                                                    
img = cv2.imread('D:\Phong\TT\sort\data\\1.jpg')
output = read_model(img)
post_process(img, output)
#print(output)