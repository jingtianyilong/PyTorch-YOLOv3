from __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def non_max_suppression(prediction, num_classes=80, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            conf_sort_index = torch.sort(detections_class[:, 4], descending=True)[1]
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output


def build_targets(
    pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim
):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    mask = torch.zeros(nB, nA, nG, nG)
    conf_mask = torch.ones(nB, nA, nG, nG)
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1
            # Convert to position relative to box
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype="uint8")[y])

# def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
#     '''
#     Perform non maximal suppression
#     :param prediction: prediction matrix
#     :param confidence: confidence threshold
#     :param num_classes: number of classes
#     :param nms_conf:
#     :return:     Returns detections with shape:(x1, y1, x2, y2, object_conf, class_score, class_pred)
#     '''
#
#     conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
#     prediction = prediction*conf_mask
#
#     box_corner = prediction.new(prediction.shape)
#     # From (center x, center y, width, height, confidence) to (x1, y1, x2, y2, confidence)
#     box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
#     box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
#     box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
#     box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
#     prediction[:,:,:4] = box_corner[:,:,:4]
#     # prediction is using x,y,w,h to describe the boundingbox, convert to the 4 corner point
#
#     batch_size = prediction.size(0)
#
#     write = False
#
#
#     for ind in range(batch_size):
#         image_pred = prediction[ind]          #image Tensor
#        #confidence threshholding
#        #NMS
#
#         max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
#         # torch.max(input) return the max value
#         # torch.max(input, dim, keepdim=False, out=None) return (max value, max value index)
#         # dim=0 search max as column, return index of the row
#         # dim =1 search max as row， return index of the column
#         max_conf = max_conf.float().unsqueeze(1) # transform +1 dimension. 3 dimension
#         max_conf_score = max_conf_score.float().unsqueeze(1)
#         seq = (image_pred[:,:5], max_conf, max_conf_score) # (5+1+1)
#         image_pred = torch.cat(seq, 1)
#         # torch.cat combine seq together. dim indicated in which direction are concatenated.
#         # dim=0 generate more row, dim=1 generate more column
#
#         non_zero_ind =  (torch.nonzero(image_pred[:,4]))
#         try:
#             image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
#             # view() convert the tensor to different size. with 7column but don't know how many rows.
#         except:
#             continue
#
#         if image_pred_.shape[0] == 0:
#             continue
# #
#
#         #Get the various classes detected in the image
#         img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index
#         # sort and remove duplicate element.
#
#         for cls in img_classes:
#             #perform NMS
#
#
#             #get the detections with one particular class
#             cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1) # filter out the object of class cls
#             class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
#             # indeces of nunzero element. output zxn. n=input_dim, z=num_nun_zero_element
#             image_pred_class = image_pred_[class_mask_ind].view(-1,7)
#
#             #sort the detections such that the entry with the maximum objectness
#             #confidence is at the top
#             conf_sort_index = torch.sort(image_pred_class[4,:], descending=True)[1]
#             print('\nsize of sonf_sort_index: ', conf_sort_index.size())
#             print('\nsize of image_pred_class: ', image_pred.size())
#             image_pred_class = image_pred_class[conf_sort_index]
#             idx = image_pred_class.size(0)   #Number of detections
#
#             for i in range(idx):
#                 #Get the IOUs of all boxes that come after the one we are looking at
#                 #in the loop
#                 try:
#                     ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
#                 except ValueError:
#                     break
#
#                 except IndexError:
#                     break
#
#                 #Zero out all the detections that have IoU > treshhold
#                 iou_mask = (ious < nms_conf).float().unsqueeze(1)
#                 image_pred_class[i+1:] *= iou_mask
#
#                 #Remove the non-zero entries
#                 non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
#                 image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
#
#             batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
#             seq = batch_ind, image_pred_class
#
#             if not write:
#                 output = torch.cat(seq,1)
#                 write = True
#             else:
#                 out = torch.cat(seq,1)
#                 output = torch.cat((output,out)) # default dim=0
#     try:
#         return output
#     except:
#         return 0

def letterbox_image(img, input_dim):
    '''resize image with unchanged aspect ratio using padding'''

    # img is a large array with rgb data
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = input_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    # calculate the actual size of the image wrt. input_dim
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    # do the resize using cv2
    
    canvas = np.full((input_dim[1], input_dim[0], 3), 128)
    # return a np array that is size(w,h,3), and filled with 128

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_image
    # resize the resized image to (416, 416, 3) np.array with gray edges filled in the blanked area
    
    return canvas

def prep_image(img, input_dim):
    """
    Prepare image for inputting to the neural network.
    
    Returns a Variable
    """

    img = (letterbox_image(img, (input_dim, input_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    # transpose (h416, w416, channel3) to (channel3, h416, w416)

    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    # div return the reletively value from original value and the setting value
    # unsqueeze add a dimension to a specific position


    return img

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np) # return an sorted array that has no duplicate element
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res
