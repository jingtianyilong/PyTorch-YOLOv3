
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import sys
import time
import torch

import pickle as pkl
from models import *
from utils.utils import *
from utils.datasets import *
from utils.kittiloader import *

import numpy as np
import random
import rospy
from sensor_msgs.msg import LaserScan
# from utils.laserSub import LaserSubs


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='config/v390.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='/home/project/ZijieMA/Trained_archiv/coco/v390_final.weights', help='path to weights file')
# parser.add_argument('--weights_path', type=str, default='weights/v390_500000.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=list, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

def write(x, results):
    c1 = tuple(x[0:2].int())
    c2 = tuple(x[2:4].int())
    img = results

    label = "{},({},{}),{:4f}".format(classes[int(x[-4])],x[-3],x[-2],x[-1])
    cv2.rectangle(img, c1, c2,(255,0,0), 4)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 5, c1[1] + t_size[1] + 6
    cv2.rectangle(img, c1, c2,(255,0,0), -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_DUPLEX, 1, [225,255,255], 1);
    return img


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

CUDA = torch.cuda.is_available() and opt.use_cuda

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
# model class Darknet(nn.Module)
model.load_weights(opt.weights_path)
input_dim = opt.img_size

point_cloud_raw = np.empty((0,2))

def callback(msg):

    global point_cloud_raw
    point_cloud_raw = np.empty((0,3))
    num_points = len(msg.ranges)
    angle_increment = msg.angle_increment
    i=0

    for point in msg.ranges:
        if point!=float('Inf'):
            # print("point!=inf")
            point_cloud_raw = np.append(point_cloud_raw,[[point*np.cos(i),point*np.sin(i),0.0]],axis=0)
        else:
            pass
        i+=angle_increment

rospy.init_node('lidarListen')
sub = rospy.Subscriber('/scan', LaserScan, callback)
# rospy.spin()

print("finish load LaserSubs")
if CUDA:
    model.cuda()

model.eval() # Set in evaluation mode

frames = 0
start_time = time.time()
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        camframes = pipeline.wait_for_frames()
        color_frame = camframes.get_color_frame()
        if not color_frame:
            continue
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        img = prep_image(color_image, input_dim) # img: torch.Tensor size([1, 3, 416, 416])
        img_dim = color_image.shape[1], color_image.shape[0] #(720 480)
        img_dim = torch.FloatTensor(img_dim).repeat(1,2) # (1 4) [720 480 720 480]
        # 重复im_dim 1行2列

        if CUDA:
            img_dim = img_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            detections = model(img)
            detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
            # write_results function performs NMS
            try:
                detections_with_distance = torch.zeros((detections[0].shape[0]), detections[0].shape[1]+3)
                detections_with_distance[:,:-3] = detections[0]

                for detection in detections_with_distance:
                    detection = get_frustum_rplidar_distance(detection, point_cloud_raw)
            except:
                pass

        if type(detections) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format(frames / (time.time() - start_time)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue

        if CUDA: detections_with_distance = detections_with_distance.cuda()
        img_dim = img_dim.repeat(detections_with_distance.size(0), 1)
        scaling_factor = torch.min(opt.img_size/img_dim,1)[0].view(-1,1)
        # view() transform the tensor in different size. in this case -1 means don't care. But column must be 1
        detections_with_distance[:,[0,2]] -= (input_dim - scaling_factor*img_dim[:,0].view(-1,1))/2
        detections_with_distance[:,[1,3]] -= (input_dim - scaling_factor*img_dim[:,1].view(-1,1))/2

        detections_with_distance[:,0:4] /= scaling_factor

        for i in range(detections_with_distance.shape[0]):
            detections_with_distance[i, [0,2]] = torch.clamp(detections_with_distance[i, [0,2]], 0.0, img_dim[i,0])
            detections_with_distance[i, [1,3]] = torch.clamp(detections_with_distance[i, [1,3]], 0.0, img_dim[i,1])

        # Clamp all elements in input into the range [ min, max ] and return a resulting tensor
        # detections = torch.cat(detections)
        # img_dim = img_dim.repeat(detections.size(0), 1)
        # scaling_factor = torch.min(opt.img_size/img_dim,1)[0].view(-1,1)
        # # view() transform the tensor in different size. in this case -1 means don't care. But column must be 1
        #
        # detections[:,[0,2]] -= (input_dim - scaling_factor*img_dim[:,0].view(-1,1))/2
        # detections[:,[1,3]] -= (input_dim - scaling_factor*img_dim[:,1].view(-1,1))/2
        #
        # detections[:,0:4] /= scaling_factor
        #
        # for i in range(detections.shape[0]):
        #     detections[i, [0,2]] = torch.clamp(detections[i, [0,2]], 0.0, img_dim[i,0])
        #     detections[i, [1,3]] = torch.clamp(detections[i, [1,3]], 0.0, img_dim[i,1])
        #     # Clamp all elements in input into the range [ min, max ] and return a resulting tensor


        classes = load_classes('data/coco.names')

        list(map(lambda x: write(x, color_image), detections_with_distance))
        cv2.imshow("frame", color_image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break


finally:
    FPS = frames/(time.time()-start_time)
    frames += 1
    print('FPS:%.04f' % FPS, end='\r')
    # Stop streaming
    pipeline.stop()
