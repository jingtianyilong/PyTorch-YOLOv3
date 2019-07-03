
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


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='config/v390.cfg', help='path to model config file')
# parser.add_argument('--weights_path', type=str, default='/home/project/ZijieMA/Trained_archiv/coco/v390_final.weights', help='path to weights file')
parser.add_argument('--weights_path', type=str, default='weights/v390_final.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.45, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=list, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

def write(x, results):
    '''
    write the image with bounding boxes and the labels
    :param x: (x,y,width,height,objectness,lable,center_x,center_y,radius)
    :param results:
    :return:
    '''
    # coordination of the bounding box
    c1 = tuple(x[0:2].int())
    # dimension of the bounding box
    c2 = tuple(x[2:4].int())
    img = results
    if classes[int(x[5])]=="person":
        label = "{},({0:.2f},{0:.2f}),{0:.2f}".format("person",x[-3],x[-2],x[-1])
        cv2.rectangle(img, c1, c2,(255,0,0), 4)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 5, c1[1] + t_size[1] + 6
        cv2.rectangle(img, c1, c2,(255,0,0), -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_DUPLEX, 1, [225,255,255], 1);
    else classes[int(x[-4])] in ('car', 'truck', 'truck'):
        label = "{},({:.2f},{:.2f}),{:.2f}".format(classes[int(x[-4])],x[-3],x[-2],x[-1])
        cv2.rectangle(img, c1, c2,(255,0,0), 4)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)[0]
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

# Set up model and read the weight file
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)
input_dim = opt.img_size

point_cloud_raw = np.empty((0,2))

def callback(msg):
    # rospy run this on the background to update point cloud information
    global point_cloud_raw
    point_cloud_raw = np.empty((0,2))
    num_points = len(msg.ranges)
    angle_increment = msg.angle_increment
    i=0

    for point in msg.ranges[0:180]:
        if point!=float('Inf'):
            point_cloud_raw = np.append(point_cloud_raw,[[np.pi-i,point]],axis=0)
        else:
            pass
        i+=angle_increment

# initial a node and subscribe to scan
rospy.init_node('lidarListen')
sub = rospy.Subscriber('/scan', LaserScan, callback)

# to make sure we use our GPU
if CUDA:
    model.cuda()
# Set to evaluation mode(so that the pytorch don't calculate the gradients)
model.eval()

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

        # resise it and change the columns
        img = prep_image(color_image, input_dim) # img: torch.Tensor size([1, 3, 416, 416])
        img_dim = color_image.shape[1], color_image.shape[0]
        img_dim = torch.FloatTensor(img_dim).repeat(1,2)
        # repeat img_dim

        # move tensore to device 'cuda' instead of 'cpu'
        if CUDA:
            img_dim = img_dim.cuda()
            img = img.cuda()

        # pridiction
        with torch.no_grad():
            detections = model(img)
            detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
            # write_results function performs NMS
            try:
                # generate 3 column to store coordination and radius
                detections_with_distance = torch.zeros((detections[0].shape[0]), detections[0].shape[1]+3)
                detections_with_distance[:,:-3] = detections[0]

                # to get coordination and radius
                for detection in detections_with_distance:
                    detection[-3:] = get_frustum_rplidar_distance(detection, point_cloud_raw)
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
        # move this to cuda
        if CUDA: detections_with_distance = detections_with_distance.cuda()
        img_dim = img_dim.repeat(detections_with_distance.size(0), 1)
        scaling_factor = torch.min(opt.img_size/img_dim,1)[0].view(-1,1)
        # view() transform the tensor in different size. in this case -1 means don't care. But column must be 1
        # convert back to original image size
        detections_with_distance[:,[0,2]] -= (input_dim - scaling_factor*img_dim[:,0].view(-1,1))/2
        detections_with_distance[:,[1,3]] -= (input_dim - scaling_factor*img_dim[:,1].view(-1,1))/2

        detections_with_distance[:,0:4] /= scaling_factor

        for i in range(detections_with_distance.shape[0]):
            detections_with_distance[i, [0,2]] = torch.clamp(detections_with_distance[i, [0,2]], 0.0, img_dim[i,0])
            detections_with_distance[i, [1,3]] = torch.clamp(detections_with_distance[i, [1,3]], 0.0, img_dim[i,1])

        # label with number to label with words
        classes = load_classes('data/coco.names')

        # draw frame
        list(map(lambda x: write(x, color_image), detections_with_distance))
        cv2.imshow("frame", color_image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break


finally:
    pipeline.stop()
