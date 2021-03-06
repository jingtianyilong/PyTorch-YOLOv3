from __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from sklearn import linear_model, datasets
from itertools import compress

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from utils.kittiloader import *
import rospy
from sensor_msgs.msg import LaserScan

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


def non_max_suppression(prediction, num_classes=80, conf_thres=0.5, nms_thres=0.45):
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

def letterbox_image(img, input_dim):
    '''resize image with unchanged aspect ratio using padding'''

    # img is a large array with rgb data
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = input_dim
    new_w = int(img_w* min(w/img_w, h/img_h))
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
    # return an sorted array that has no duplicate element
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def get_frustum_point_distance(img_id, img_path, detection, kitti_path, img_size_after_resize):
    detection = detection.numpy()
    img_id = int(img_path[0][-10:-4])
    lidar_path = '%straining/velodyne/%06d.bin' % (kitti_path, img_id)
    calib = calibread('%straining/calib/%06d.txt' % (kitti_path, img_id))
    img = cv2.imread('/home/project/ZijieMA/PyTorch-YOLOv3/examples/%06d.png' % img_id, cv2.IMREAD_UNCHANGED)
    # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img_width_orig = img.shape[1]

    img_height_orig = img.shape[0]

    pad_x = max(img_height_orig - img_width_orig, 0) * (max(img_size_after_resize) / max(img_width_orig, img_height_orig))
    pad_y = max(img_width_orig - img_height_orig, 0) * (max(img_size_after_resize) / max(img_width_orig, img_height_orig))
    # Image height and width after padding is removed
    unpad_h = img_size_after_resize[1] - pad_y
    unpad_w = img_size_after_resize[0] - pad_x
    box_h = ((detection[3] - detection[1]) / unpad_h) * img_height_orig
    box_w = ((detection[2] - detection[0]) / unpad_w) * img_width_orig
    v_upper = ((detection[1] - pad_y // 2) / unpad_h) * img_height_orig
    u_left = ((detection[0] - pad_x // 2) / unpad_w) * img_width_orig
    v_bottom = v_upper + box_h
    u_right = u_left + box_w

    point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    # orig_point_cloud = point_cloud # nx4

    # detections with shape: (x1, y1, x2, y2, object_conf, class_score, class_pred)

    ########################################################################
    # Distance rough estimation
    # D = H [tan(theta_c+arctan((h_i/2-d_p)/(h/(2*tan(alpha/2)))))-tan(theta_c-alpha/2)]
    # reference：
    # Computer_Vision_for_Road_Safety_A_System
    # H         height of the camera(according to kitti is 1.65m)
    # alpha     angle of FOV in v-axis fv=h_i/(2*tan(alpha/2))
    # theta_c   angle between camera x-axis and X-axis(pi/2)
    # h_i       height of the recorded image plane(pixel)(512)
    # d_p       distance from the bottom of image to the bottom of the bounding box(512-v_min)
    #
    # D = H ×[tan（theta_c+arctan((h_i-d_p)/fv))-tan(theta_c-arctan(hi/(2fv)))]
    #
    ########################################################################


    # # # # # debug visualization:
    # pcd = PointCloud()
    # pcd.points = Vector3dVector(point_cloud[:, 0:3])
    # pcd.paint_uniform_color([0.65, 0.65, 0.65])
    # draw_geometries_dark_background([pcd])
    # 362 252 207 214
    # # # # #

    P2 = calib["P2"] # 3x4 matris projection matrix after rectification
    # （u,v,1） = dot(P2, (x,y,z,1))
    Height_of_camera = 1.65
    fu = P2[0][0]  # for horizontal position
    fv = P2[1][1]
    # theta_c = np.pi/2
    D_rough = Height_of_camera * fv / (v_bottom - img_height_orig/2)
    # D_rough = Height_of_camera * (np.tan(theta_c + np.arctan((img_height_orig/2 - d_p)/fv)) - np.tan(theta_c - np.arctan(img_height_orig/(2*fv))))
    print(D_rough)
    if D_rough > 0:
        # remove points that are located behind the camera:
        point_cloud = point_cloud[point_cloud[:, 0] > (D_rough - 2), :]
        # remove points that are located too far away from the camera:
        point_cloud = point_cloud[point_cloud[:, 0] < min(80, D_rough + 2), :]
        point_cloud = point_cloud[point_cloud[:,2] > Height_of_camera,:]


        Tr_velo_to_cam_orig = calib["Tr_velo_to_cam"]
        R0_rect_orig = calib["R0_rect"] # 3x3

        R0_rect = np.eye(4)
        R0_rect[0:3, 0:3] = R0_rect_orig # 3x3 -> 4x4 up left corner
        ########################################################################
        # R0_rect: example
        # array([[ 0.99, 0.01, 0.01,   0 ],
        #        [ 0.01, 0.99, 0.01,   0 ],
        #        [ 0.01, 0.01, 0.99,   0 ],
        #        [    0,    0,    0,   1 ]])
        ########################################################################

        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig # 3x4 -> 4x4 up left corner
        ########################################################################
        # Tr_velo_to_cam:
        # Tr_velo_to_cam = [ R_velo_to_cam,    t_velo_to_cam ]
        #                  [             0,                1 ]
        # Rotation matrix velo -> camera 3x3, translation vector velo ->camera 1x3
        ########################################################################

        point_cloud_xyz = point_cloud[:, 0:3] # num_point x 3 (x,y,z,reflectance) reflectance don't need
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))
        # the 4th column are all 1

        # project the points onto the image plane (homogeneous coords):
        img_points_hom = np.dot(P2, np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T))).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # (U,V,_) = P2 * R0_rect * Tr_velo_to_cam * point_cloud_xyz_hom
        # normalize: (U,V,1)
        img_points = np.zeros((img_points_hom.shape[0], 2))
        img_points[:, 0] = img_points_hom[:, 0]/img_points_hom[:, 2]
        img_points[:, 1] = img_points_hom[:, 1]/img_points_hom[:, 2]

        # transform the points into (rectified) camera coordinates:
        point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize: (x,y,z,1)
        point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
        point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]

        point_cloud_camera = point_cloud
        point_cloud_camera[:, 0:3] = point_cloud_xyz_camera # reserve reflection

        ########################################################################
        # point_cloud               n x 4   original xyzr value before cali in velo coordinate
        # point_cloud_xyz           n x 3   xyz value before cali in velo coordinate
        # point_cloud_xyz_hom       n x 4   xyz1 in velo coordinate
        # point_cloud_xyz_camera    n x 4   xyz1 in camera coordinate
        # point_cloud_camera        n x 4   xyzr in camera coordinate
        # img_points_hom            n x 3   uv_
        # img_points                n x 2   UV
        ########################################################################


        row_mask = np.logical_and(
                        np.logical_and(img_points[:, 0] >= u_left,
                                       img_points[:, 0] <= u_right),
                        np.logical_and(img_points[:, 1] >= v_upper,
                                       img_points[:, 1] <= v_bottom))

        # filter out point are not in frustum area
        frustum_point_cloud_xyz = point_cloud_xyz[row_mask, :] # (needed only for visualization)
        frustum_point_cloud = point_cloud[row_mask, :]
        frustum_point_cloud_xyz_camera = point_cloud_xyz_camera[row_mask, :]
        frustum_point_cloud_camera = point_cloud_camera[row_mask, :]

        # randomly sample 512 points in the frustum point cloud:


        if frustum_point_cloud.shape[0] == 0:
             detection[7] = D_rough
             return torch.tensor(detection)
        # elif frustum_point_cloud.shape[0] < 512:
        #     row_idx = np.random.choice(frustum_point_cloud.shape[0], 512, replace=True)
        # else:
        #     row_idx = np.random.choice(frustum_point_cloud.shape[0], 512, replace=False)

        frustum_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_idx, :]
        ransac = linear_model.RANSACRegressor()
        ransac.fit(frustum_point_cloud_xyz_camera[:,1].reshape(-1,1),frustum_point_cloud_xyz_camera[:,0].reshape(-1,1))

        right_side_distance = ransac.predict([[frustum_point_cloud_xyz_camera[:,1].max()]])[0][0]
        left_side_distance = ransac.predict([[frustum_point_cloud_xyz_camera[:,1].min()]])[0][0]

        detection[7] = min(min(left_side_distance,right_side_distance),D_rough-2)
        print('image id:', img_id)
        print('Rough estimation %d, \n ransac estimation: %d %d, \n final estimation: %d' %(rough_D,left_side_distance,right_side_distance,detection[7]))
        return torch.tensor(detection)

    else:# might be a problem
        detection[7] = float('nan')
        return torch.tensor(detection)

def get_frustum_point_distance_simplified(img_id, img_path, detection, kitti_path, img_size_after_resize):
    '''
    ###############################################################################################
    Only return distance(in x_direction)
    if you want the contour estimation, go to visualization_ipynote/Ransac with Edges.ipynb
    ###############################################################################################
    :param img_id: dataset image id on kitti
    :param img_path: used to read the image and check the size(if the image are all in same size we can eliminate this)
    :param detection: bounding box information from CNN
    :param kitti_path: to joint the path
    :param img_size_after_resize: input size of the CNN
    :return:
    '''

    detection = detection.numpy()

    # read information about point cloud, image and the calibration data
    img_id = int(img_path[0][-10:-4])
    lidar_path = '%straining/velodyne/%06d.bin' % (kitti_path, img_id)
    calib = calibread('%straining/calib/%06d.txt' % (kitti_path, img_id))
    img = cv2.imread('/home/project/ZijieMA/PyTorch-YOLOv3/examples/%06d.png' % img_id, cv2.IMREAD_UNCHANGED)
    point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

    # calculate the bounding box information in original size
    box_h = ((detection[3] - detection[1]) / unpad_h) * img_height_orig
    box_w = ((detection[2] - detection[0]) / unpad_w) * img_width_orig
    v_upper = ((detection[1] - pad_y // 2) / unpad_h) * img_height_orig
    u_left = ((detection[0] - pad_x // 2) / unpad_w) * img_width_orig
    v_bottom = v_upper + box_h
    u_right = u_left + box_w

    # rough estimation if the camera is front facing
    D_rough = Height_of_camera * fv / (v_bottom - img_height_orig / 2)
    if D_rough > 0:
        # remove points that are located behind the camera:
        point_cloud = point_cloud[point_cloud[:, 0] > (D_rough - 3), :]
        # remove points that are located too far away from the camera:
        point_cloud = point_cloud[point_cloud[:, 0] < min(80, D_rough + 3), :]
        # remove points that are above and under certain height (make sure the contour of the vehicle is L-shape)
        point_cloud = point_cloud[point_cloud[:, 2] > -1.5, :]
        point_cloud = point_cloud[point_cloud[:, 2] < -1, :]

        ########################################################################
        # point_cloud               n x 4   original xyzr value before cali in velo coordinate
        # point_cloud_xyz_hom       n x 4   xyz1 in velo coordinate
        # img_points_hom            n x 3   uv_
        # img_points                n x 2   UV
        ########################################################################

        # those actually transform all the points to the camera0 coordination.(we use camera2)
        R0_rect = np.eye(4)
        R0_rect[0:3, 0:3] = calib["R0_rect"]  # 3x3 -> 4x4 up left corner
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[0:3, :] = calib["Tr_velo_to_cam"]  # 3x4 -> 4x4 up left corner

        # remove reflection information from point_cloud
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3]  # (point_cloud_xyz_hom has shape (num_points, 4))
        # the 4th column are all 1

        # project the points onto the image plane (homogeneous coords):
        # (U,V,_) = P2 * R0_rect * Tr_velo_to_cam * point_cloud_xyz_hom
        # normalize: (U,V,1)
        img_points_hom = np.dot(P2, np.dot(R0_rect, np.dot(Tr_velo_to_cam,
                                                           point_cloud_xyz_hom.T))).T  # (point_cloud_xyz_hom.T has shape (4, num_points))
        img_points = np.zeros((img_points_hom.shape[0], 2))
        img_points[:, 0] = img_points_hom[:, 0] / img_points_hom[:, 2]
        img_points[:, 1] = img_points_hom[:, 1] / img_points_hom[:, 2]

        # index of the points in the
        row_mask = np.logical_and(
            np.logical_and(img_points[:, 0] >= u_left,
                           img_points[:, 0] <= u_right),
            np.logical_and(img_points[:, 1] >= v_upper,
                           img_points[:, 1] <= v_bottom))

        if frustum_point_cloud.shape[0] == 0:
             detection[7] = D_rough
             return torch.tensor(detection)

        frustum_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_idx, :]
        ransac = linear_model.RANSACRegressor()
        ransac.fit(frustum_point_cloud_xyz_camera[:,1].reshape(-1,1),frustum_point_cloud_xyz_camera[:,0].reshape(-1,1))

        right_side_distance = ransac.predict([[frustum_point_cloud_xyz_camera[:,1].max()]])[0][0]
        left_side_distance = ransac.predict([[frustum_point_cloud_xyz_camera[:,1].min()]])[0][0]

        detection[7] = min(min(left_side_distance,right_side_distance),D_rough-2)
        print('image id:', img_id)
        print('Rough estimation %d, \n ransac estimation: %d %d, \n final estimation: %d' %(rough_D,left_side_distance,right_side_distance,detection[7]))
        return torch.tensor(detection)

    else:# might be a problem
        detection[7] = float('nan')
        return torch.tensor(detection)


def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    '''generate intersection point using 4 different point'''
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return px, py

def findEdgePoint(x1, y1, x2, y2, x3, y3, x4, y4):
    '''
    checking point that are on the far side given 2 lines described using 4 points.
    '''
    df = [[x1,y1,0],[x2,y2,0],[x3,y3,0],[x4,y4,0]]
#     for point in df:
#         point[2]=point[0]**2+point[1]**2
    df = sorted(df,key=lambda x: x[0]**2+x[1]**2,reverse=True)
    # point start always from left side
    if df[0][1] <= df[1][1]:
        return df[0][0], df[0][1], df[1][0], df[1][1]
    else:
        return df[1][0], df[1][1], df[0][0], df[1][1]

def ransac_with_bbox(point_cloud_with_mask, category,rough_D):
    '''
    designed for KITTI detection
    :param point_cloud_with_mask: point cloud in frustum area
    :param n_sample: sample to start ransac(should be less than len(point_cloud_with_mask))
    :param category: decide how we deal with the regresion
    '''
    # category numbers are according to the datasets. Here with COCO
    # car, bus and truck
    if category in [2,5,7]:
        '''
        idea with the car is to use ransac to filter out outliers and regression with one line that describes a
        edge. The outliers then run another ransac to generate another edge
        '''
        # points that are have distance larger than 0.1 will regard as outlier
        ransac = linear_model.RANSACRegressor(residual_threshold=0.1)
        ransac.fit(point_cloud_with_mask[:, 1].reshape(-1, 1), point_cloud_with_mask[:, 0].reshape(-1, 1))

        # line points for first ransac
        inlier_index = list(compress(range(len(ransac.inlier_mask_)), ransac.inlier_mask_))
        # side vertex for each line
        left_y_1 = point_cloud_with_mask[inlier_index, 1].min()
        right_y_1 =  point_cloud_with_mask[inlier_index, 1].max()
        left_x_1 = ransac.predict([[left_y_1, ]])[0][0]
        right_x_1 = ransac.predict([[right_y_1, ]])[0][0]

        # list out all the outlier
        outlier_index = list(compress(range(len(ransac.inlier_mask_)), [not i for i in ransac.inlier_mask_]))
        outlier_point_cloud = point_cloud_with_mask[outlier_index,:]

        # run the second ransac only when outlier points are more than 0.5
        if len(outlier_index)>1/5*(len(point_cloud_with_mask)):

            ransac.fit(outlier_point_cloud[:, 1].reshape(-1, 1),
                       outlier_point_cloud[:, 0].reshape(-1, 1))

            inlier_index = list(compress(range(len(ransac.inlier_mask_)), ransac.inlier_mask_))
            left_y_2 = outlier_point_cloud[inlier_index, 1].min()
            right_y_2 =  outlier_point_cloud[inlier_index, 1].max()
            left_x_2 = ransac.predict([[left_y_2, ]])[0][0]
            right_x_2 = ransac.predict([[right_y_2, ]])[0][0]

            # intersection point of two ransec result
            x_2, y_2 = findIntersection(left_x_1,left_y_1,
                                        right_x_1,right_y_1,
                                        left_x_2,left_y_2,
                                        right_x_2,right_y_2)
            x_1, y_1, x_3, y_3 = findEdgePoint(left_x_1,left_y_1,
                                               right_x_1,right_y_1,
                                               left_x_2,left_y_2,
                                               right_x_2,right_y_2)

            # return coordination of 3 contour point, from left to right.
            # make sure the intersection point is in the area
            if D_rough-3 <x_2< D_rough +3 and point_cloud_mask[:,1].min()<y_2<point_cloud_mask[:,1].max():
                return [x_1, y_1, x_2, y_2, x_3, y_3]
            else: [left_x_1, left_y_1, right_x_1, right_y_1]
        else: return [left_x_1, left_y_1, right_x_1, right_y_1]

    # person: return with center position(x,y) and radius r
    elif category == 0:
        '''the idea of the human can use the center cluster to be the distance'''

        radius = 6*np.std(point_cloud_mask[:,1])
        center_x = np.median(point_cloud_mask[:,0])+0.2
        center_y = np.median(point_cloud_mask[:,1])
        return [center_x, center_y,radius]
    else:
        return None

def get_frustum_rplidar_distance(detection, point_cloud):
    '''designed for the abschluss demo
    only for human detection
    without rough estimation using image
    without all those matrix to find the frustum region
    '''
    detection = detection.numpy()

    '''  resolutionRGB: 1920 1080
  FocalLengthColor: 1387.531128 1387.352051
  PrincipalPointColor: 956.005127 558.462280
  DistortionColor: 0.000000 0.000000 0.000000 0.000000 0.000000
  RotationLeftColor: 0.999866 -0.014898 0.006766
                     0.014936 0.999873 -0.005493
                     -0.006683 0.005593 0.999962
  TranslationLeftColor: 14.999831 0.375140 0.036474'''




    if detection[-4] in (0, 2, 5, 7): # make sure it is a person or vehicle
        point_cloud = point_cloud.reshape(-1, 2)

        img_width_orig = 1920 # image width
        img_height_orig = 1080
        Height_of_camera = 1.0

        fv = 1387.352051
        fu = 1387.531128 # focal length in u_coordination

        unpad_w = 416 # detection information from CNN is with this size
        unpad_h = 416

        pad_y = unpad_w * (1-fv/fu)
        # calculate the bounding box information in original size
        box_w = ((detection[2] - detection[0]) / unpad_w) * img_width_orig
        u_left = (detection[0] / unpad_w) * img_width_orig
        box_h = ((detection[3] - detection[1]) / unpad_h) * img_height_orig
        v_upper = ((detection[1] - pad_y // 2) / unpad_h) * img_height_orig
        v_bottom = v_upper + box_h

        # rough estimation if the camera is front facing
        D_rough = Height_of_camera * fv / (v_bottom - img_height_orig / 2)

        # calculate the angle to the side of the bounding box
        angle_left = np.arctan2(u_left - 0.5 * img_width_orig, fu) + np.pi/2
        angle_right = np.arctan2((u_left + box_w) - 0.5* img_width_orig, fu) + np.pi/2
        row_mask = np.logical_and(point_cloud[:,0] >= angle_left, point_cloud[:,0] <= angle_right)

        if len(row_mask)!=0:
            point_cloud_mask = np.empty((0,2))

            for point in point_cloud[row_mask]:
                point_cloud_mask = np.append(point_cloud_mask,[[point[1]*np.cos(point[0]),point[1]*np.sin(point[0])]],axis=0)


            if detection[-4] == 0:
                # coordination and radius of the detected human
                center_x = np.median(point_cloud_mask[:,1])+0.2
                center_y = np.median(point_cloud_mask[:,0])
                radius = box_w/fu * center_x

                return torch.tensor([center_x, center_y, radius])

            elif detection[5] in (2, 5, 7):

                ransac = linear_model.RANSACRegressor(residual_threshold=0.1)
                ransac.fit(point_cloud_mask[:, 1].reshape(-1, 1), point_cloud_mask[:, 0].reshape(-1, 1))

                # line points for first ransac
                inlier_index = list(compress(range(len(ransac.inlier_mask_)), ransac.inlier_mask_))
                # side vertex for each line
                left_y_1 = point_cloud_mask[inlier_index, 1].min()
                right_y_1 = point_cloud_mask[inlier_index, 1].max()
                left_x_1 = ransac.predict([[left_y_1, ]])[0][0]
                right_x_1 = ransac.predict([[right_y_1, ]])[0][0]

                # list out all the outlier
                outlier_index = list(compress(range(len(ransac.inlier_mask_)), [not i for i in ransac.inlier_mask_]))
                outlier_point_cloud = point_cloud_mask[outlier_index, :]

                # run the second ransac only when outlier points are more than 0.5
                if len(outlier_index) > 1 / 5 * (len(point_cloud_mask)):

                    ransac.fit(outlier_point_cloud[:, 1].reshape(-1, 1),
                               outlier_point_cloud[:, 0].reshape(-1, 1))

                    inlier_index = list(compress(range(len(ransac.inlier_mask_)), ransac.inlier_mask_))
                    left_y_2 = outlier_point_cloud[inlier_index, 1].min()
                    right_y_2 = outlier_point_cloud[inlier_index, 1].max()
                    left_x_2 = ransac.predict([[left_y_2, ]])[0][0]
                    right_x_2 = ransac.predict([[right_y_2, ]])[0][0]

                    # intersection point of two ransec result
                    x_2, y_2 = findIntersection(left_x_1, left_y_1,
                                                right_x_1, right_y_1,
                                                left_x_2, left_y_2,
                                                right_x_2, right_y_2)
                    x_1, y_1, x_3, y_3 = findEdgePoint(left_x_1, left_y_1,
                                                       right_x_1, right_y_1,
                                                       left_x_2, left_y_2,
                                                       right_x_2, right_y_2)

                    # return coordination of 3 contour point, from left to right.
                    # make sure the intersection point is in the area
                    if D_rough - 3 < x_2 < D_rough + 3 and point_cloud_mask[:, 1].min() < y_2 < point_cloud_mask[:,
                                                                                                1].max():
                        return torch.tensor([x_1, y_1, x_2, y_2, x_3, y_3])
                    else:
                        return torch.tensor([left_x_1, left_y_1, right_x_1, right_y_1])
                else:
                    return torch.tensor([left_x_1, left_y_1, right_x_1, right_y_1])
        else:
            return torch.tensor([0, 0, 0])

    else:
        return torch.tensor([0, 0, 0])
