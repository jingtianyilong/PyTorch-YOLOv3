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
parser.add_argument('--image_folder', type=str, default='data/samples/samples/', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/v390.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='weights/v390_280000.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

def write(x, results):
    c1 = tuple(x[0:2].int())
    c2 = tuple(x[2:4].int())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,(255,0,0), 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,(255,0,0), 2)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

CUDA = torch.cuda.is_available() and opt.use_cuda
# os.makedirs('output', exist_ok=True)

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
# model class Darknet(nn.Module)
model.load_weights(opt.weights_path)
input_dim = opt.img_size

if CUDA:
    model.cuda()

model.eval() # Set in evaluation mode

video_file = '/home/zijieguo/project/darknet/IMG_8765.mov'
cap = cv2.VideoCapture(video_file)
# cap = cv2.VideoCapture(0)


assert cap.isOpened(), 'failed to load camera video'

frames = 0
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        img = prep_image(frame, input_dim) # img: torch.Tensor size([1, 3, 416, 416])
#        cv2.imshow("a", frame)
        img_dim = frame.shape[1], frame.shape[0] #(720 480)
        img_dim = torch.FloatTensor(img_dim).repeat(1,2) # (1 4) [720 480 720 480]
        # 重复im_dim 1行2列

        if CUDA:
            img_dim = img_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            detections = model(img)
            # detections = write_results(detections,opt.conf_thres, 80, nms_conf = opt.nms_thres)
            detections = non_max_suppression(detections)
        # write_results function performs NMS


        if type(detections) == int:
            frames += 1
        #     print("FPS of the video is {:5.4f}".format(frames / (time.time() - start_time)))
        #     cv2.imshow("frame", frame)
        #     key = cv2.waitKey(1)
        #     if key & 0xFF == ord('q'):
        #         break
        #     continue

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
        # classes = load_classes('data/coco.names')
        # colors = pkl.load(open("pallete", "rb"))
        #
        # list(map(lambda x: write(x, frame), detections))
        #
        # cv2.imshow("frame", frame)
        # key = cv2.waitKey(1)
        # if key & 0xFF == ord('q'):
        #     break
        frames += 1
        print("FPS of the video is {:5.2f}".format(FPS))
    else:
        break

