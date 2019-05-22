from PIL import Image
from os.path import isfile, join
import numpy as np
import os
import random


def IOU(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape`
    return np.array(similarities)


def avg_IOU(X, centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum += max(IOU(X[i], centroids))
    return sum / n


def write_anchors_to_file(centroids, X, anchor_file):
    f = open(anchor_file, 'w')

    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0] *= width_in_cfg_file / 32.
        anchors[i][1] *= height_in_cfg_file / 32.

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, ' % (anchors[i, 0], anchors[i, 1]))

    # there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n' % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1]))

    f.write('%f\n' % (avg_IOU(X, centroids)))
    print()


def kmeans(X, centroids, anchor_file):
    N = X.shape[0]
    iterations = 0
    k, dim = centroids.shape
    prev_assignments = np.ones(N) * (-1)
    iter = 0
    old_D = np.zeros((N, k))

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)

        print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_D - D))))

        # assign samples to centroids f
        assignments = np.argmin(D, axis=1)

        if (assignments == prev_assignments).all():
            print("Centroids = ", centroids)
            write_anchors_to_file(centroids, X, anchor_file)
            return

        # calculate new centroids
        centroid_sums = np.zeros((k, dim), np.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j))

        prev_assignments = assignments.copy()
        old_D = D.copy()
        global counter
        print(counter)
        counter += 1

file_list = open('/home/project/ZijieMA/darknet/data/coco/trainvalno5k.txt')
box_widths = []
box_heights = []
for line in file_list.readlines():
    line = line.rstrip('\n')
    label_dir = line[0:40] + "labels" + line[46:-3] + "txt"
    image_dir = line
    im = Image.open(image_dir)
    img_width, img_height = im.size
    label_handle = open(label_dir)

    for label in label_handle.readlines():
        value = label.rstrip('\n').split(' ')
        box_widths.append(float(value[3]) * 416)
        box_heights.append(float(value[4]) * 416)

X = list(zip(box_widths,box_heights))

width_in_cfg_file = 416.
height_in_cfg_file = 416.
output_dir = 'generated_anchor/'
num_clusters = 9

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

counter = 0

annotation_dims = []

size = np.zeros((1,1,3))

annotation_dims = np.array(X)


anchor_file = join(output_dir,'anchors%d.txt'%(num_clusters))
indices = [ random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
centroids =  annotation_dims[indices]
kmeans(annotation_dims,centroids,anchor_file)
print('centroids.shape', centroids.shape)