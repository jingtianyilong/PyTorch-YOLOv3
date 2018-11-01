import cv2
import argparse
import time
import torch

import pickle as pkl
from models import *
from utils.utils import *

import numpy as np
import random

print('start program')
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='config/v390.cfg', help='path to model config file')
# parser.add_argument('--weights_path', type=str, default='/home/zijieguo/project/darknet/yolov3.weights', help='path to weights file')
parser.add_argument('--weights_path', type=str, default='weights/v390_500000.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
opt = parser.parse_args()


# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
# model class Darknet(nn.Module)
model.load_weights(opt.weights_path)
input_dim = opt.img_size
model.cuda()
model.eval() # Set in evaluation mode

cap = cv2.VideoCapture('/home/zijieguo/project/darknet/IMG_8765.mov')


assert cap.isOpened(), 'failed to load camera video'

frames = 0
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        img = prep_image(frame, input_dim).cuda() # img: torch.Tensor size([1, 3, 416, 416])
        img_dim = frame.shape[1], frame.shape[0] #(720 480)
        img_dim = torch.FloatTensor(img_dim).repeat(1,2).cuda() # (1 4) [720 480 720 480]
        # 重复im_dim 1行2列

        with torch.no_grad():
            detections = model(img)
            detections = non_max_suppression(detections,80,opt.conf_thres)
        # write_results function performs NMS

        if type(detections) == int:
            frames += 1

        detections = torch.cat(detections)
        img_dim = img_dim.repeat(detections.size(0), 1)
        scaling_factor = torch.min(opt.img_size/img_dim,1)[0].view(-1,1)
        # view() transform the tensor in different size. in this case -1 means don't care. But column must be 1

        detections[:,[0,2]] -= (input_dim - scaling_factor*img_dim[:,0].view(-1,1))/2
        detections[:,[1,3]] -= (input_dim - scaling_factor*img_dim[:,1].view(-1,1))/2
        detections[:,0:4] /= scaling_factor

        for i in range(detections.shape[0]):
            detections[i, [0,2]] = torch.clamp(detections[i, [0,2]], 0.0, img_dim[i,0])
            detections[i, [1,3]] = torch.clamp(detections[i, [1,3]], 0.0, img_dim[i,1])
            # Clamp all elements in input into the range [ min, max ] and return a resulting tensor

        FPS = frames/(time.time()-start_time)
        frames += 1

    else:
        print("FPS of the video is {:5.2f}".format(FPS))
        break
