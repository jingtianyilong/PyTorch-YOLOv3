# from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.kittiloader import *


import os
import sys
import time
import datetime
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import time

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='examples/', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/v390.cfg', help='path to model config file')
parser.add_argument('--weights_path', type  =str, default='weights/v390_final.weights', help='path to weights file')
parser.add_argument('--kitti_path', type=str, default='/home/project/ZijieMA/KITTI/', help='path to kitti path')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=list, default=[416,416], help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

CUDA_available = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)

model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)
input_dim = opt.img_size

if CUDA_available:
    model.cuda()

model.eval()

# dataloader = DataLoader(ImageFolder( opt.kitti_path + 'testing/image_2', img_size=opt.img_size),
                        # batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

classes = load_classes(opt.class_path)
# TODO: different class file for 

Tensor = torch.cuda.FloatTensor if CUDA_available else torch.FloatTensor


imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index
start_time = time.time()
print('starting time: ', start_time)
print ('\nPerforming object detection:')
prev_time = time.time()
inference_time = datetime.timedelta(seconds=prev_time - prev_time)
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):

    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))
    # Viriable API has been deprecated. Viriable(tensor) return Tensors instead of Variable
    # transform input_imgs to Tensor

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs) # size 1x10647x85
        detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
        # 有一些任务，可能事先需要设置，事后做清理工作。对于这种场景，Python的with语句提供了一种非常方便的处理方式。
        # 一个很好的例子是文件处理，你需要获取一个文件句柄，从文件中读取数据，然后关闭文件句柄。基本思想是with所求值
        # 的对象必须有一个__enter__()方法，一个__exit__()方法。紧跟with后面的语句被求值后，返回对象的__enter__()
        # 方法被调用，这个方法的返回值将被赋值给as后面的变量。当with后面的代码块全部被执行完之后，将调用前面返回对象的
        # __exit__()方法。
        # similar to: try: handle = open(file) ; ...; finally: handle.close()

    # detections = torch.cat(detections)

    #  TODO: img_index -> lidar_index -> lidar filter.
    # add one column to for distance of the object
    # detections_with_distance = np.zeros((detections[0].shape[0],detections[0].shape[1]+1))
    if detections[0] is not None:
        detections_with_distance = torch.zeros((detections[0].shape[0],detections[0].shape[1]+1))

        detections_with_distance[:,:-1] = detections[0]

        for detection in detections_with_distance:
            detection = get_frustum_point_distance(batch_i, img_paths, detection, opt.kitti_path, opt.img_size)

        detections[0] = detections_with_distance
    # Log progress
    current_time = time.time()
    inference_time += datetime.timedelta(seconds=current_time - prev_time)
    # print('\t+ Batch %d, Inference Time: %s, reletively s' % (batch_i, inference_time, 1/(inference_time.total_seconds())))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)
    prev_time = current_time

# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
inference_time = inference_time/batch_i
#
# print ('\nSaving images:')
# # Iterate through images and save plot of detections
# for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
#
#     print ("(%d) Image: '%s'" % (img_i, path))
#
#     # Create plot
#     img = np.array(Image.open(path))
#     plt.figure()
#     fig, ax = plt.subplots(1)
#     ax.imshow(img)
#
#     # The amount of padding that was added
#     pad_x = max(img.shape[0] - img.shape[1], 0) * (max(opt.img_size) / max(img.shape))
#     pad_y = max(img.shape[1] - img.shape[0], 0) * (max(opt.img_size) / max(img.shape))
#     # Image height and width after padding is removed
#     unpad_h = opt.img_size[1] - pad_y
#     unpad_w = opt.img_size[0] - pad_x
#
#     # Draw bounding boxes and labels of detections
#     if detections is not None:
#         unique_labels = detections[:, -2].cpu().unique()
#         n_cls_preds = len(unique_labels)
#         bbox_colors = random.sample(colors, n_cls_preds)
#         for x1, y1, x2, y2, conf, cls_conf, cls_pred, distance in detections:
#
#             print ('\t+ Label: %s, Conf: %.5f  Dist: %.5f' % (classes[int(cls_pred)], cls_conf.item(), distance))
#             # Rescale coordinates to original dimensions
#             box_h = ((y2 - y1) / unpad_h) * img.shape[0]
#             box_w = ((x2 - x1) / unpad_w) * img.shape[1]
#             y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
#             x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
#
#             color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
#             # Create a Rectangle patch
#             bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
#                                     edgecolor=color,
#                                     facecolor='none')
#             # Add the bbox to the plot
#             ax.add_patch(bbox)
#             # Add label
#             plt.text(x1, y1, s=classes[int(cls_pred)] + ', %.5f' % distance, color='white', verticalalignment='top',
#                     bbox={'color': color, 'pad': 0})
#
#     # Save generated image with detections
#     plt.axis('off')
#     plt.gca().xaxis.set_major_locator(NullLocator())
#     plt.gca().yaxis.set_major_locator(NullLocator())
#     plt.savefig('output/%d.png' % (img_i),dpi=300, bbox_inches='tight', pad_inches=0.0)
#     plt.close()
print('processing time: %s s, which means %f FPS' %(inference_time, len(imgs)/int(inference_time)))
