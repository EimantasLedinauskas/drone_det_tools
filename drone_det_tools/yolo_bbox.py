from drone_det_tools.fake_imgs import random_insert
from drone_det_tools.misc import load_imgs
from drone_det_tools.predictions_analysis import iou

import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from keras.utils import Sequence


def bboxes_to_Y(bboxes, img_shape, grid_shape):
    Y = np.zeros((*grid_shape, 5), dtype='float32')
    grid_l_x = img_shape[1] / grid_shape[1]
    grid_l_y = img_shape[0] / grid_shape[0]
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x = 0.5 * (x1 + x2)
        y = 0.5 * (y1 + y2)
        w = x2 - x1
        h = y2 - y1
        grid_j = int(x / grid_l_x)
        grid_i = int(y / grid_l_y)
        Y[grid_i, grid_j, 0] = 1.0
        Y[grid_i, grid_j, 1] = x / grid_l_x - grid_j
        Y[grid_i, grid_j, 2] = y / grid_l_y - grid_i
        Y[grid_i, grid_j, 3] = w / grid_l_x
        Y[grid_i, grid_j, 4] = h / grid_l_y

    return Y


def bboxes_from_Y(Y, img_shape, threshold):
    bboxes = []  # x1, y1, x2, y2
    confs = []  # confidences
    grid_shape = Y.shape[0:2]
    grid_l_x = img_shape[1] / grid_shape[1]
    grid_l_y = img_shape[0] / grid_shape[0]

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if Y[i, j, 0] > threshold:
                x = (Y[i, j, 1] + j) * grid_l_x
                y = (Y[i, j, 2] + i) * grid_l_y
                w = Y[i, j, 3] * grid_l_x
                h = Y[i, j, 4] * grid_l_y
                x1 = x - w // 2
                y1 = y - h // 2
                x2 = x + w // 2
                y2 = y + h // 2
                bboxes.append((x1, y1, x2, y2))
                confs.append(Y[i, j, 0])

    return np.array(bboxes), np.array(confs)


def random_img(imgs):
    idx = np.random.choice(len(imgs))
    return imgs[idx]


class BboxFakeGenerator(Sequence):

    colors = {'color': 0, 'grayscale': 1, 'thermal': 2}

    def __init__(self, bgr_paths, drone_paths, batch_size, batches_per_epoch,
                 size_range, rot_range, img_shape, grid_shape,
                 bird_paths=None, bgr_augmenter=None, augmenter=None, colors='color'):

        self.bgr_imgs = load_imgs(bgr_paths, img_shape)
        self.drone_imgs = load_imgs(drone_paths, alpha=True)
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.size_range = size_range
        self.rot_range = rot_range
        self.img_shape = img_shape
        self.grid_shape = grid_shape
        self.bird_imgs = None if bird_paths is None else load_imgs(bird_paths, alpha=True)
        self.bgr_augmenter = bgr_augmenter
        self.augmenter = augmenter
        self.color = CoordFakeGenerator.colors[colors]

        self.on_epoch_end()

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):

        n_channels = 1 if self.color in (1, 2) else 3
        X = np.empty((self.batch_size, *self.img_shape, n_channels), dtype='float32')
        if type(self.grid_shape) is list:
            Y = [np.empty((self.batch_size, *shape, 5), dtype='float32') for shape in self.grid_shape]
        else:
            Y = np.empty((self.batch_size, *self.grid_shape, 5), dtype='float32')

        for i_batch in range(self.batch_size):

            bgr_img = random_img(self.bgr_imgs)
            if self.bgr_augmenter is not None:
                bgr_img = self.bgr_augmenter.augment_image(bgr_img)

            if self.bird_imgs is not None:
                n_birds = np.random.choice(4)
                for i in range(n_birds):
                    bird_img = random_img(self.bird_imgs)
                    bgr_img, _ = random_insert(bgr_img, bird_img, self.size_range, self.rot_range,
                                                  uniform=False, thermal=self.color==2)

            n_drones = np.random.choice(range(1,4))
            bboxes = np.empty((n_drones, 4))  # bboxes or coords
            for i in range(n_drones):
                drone_img = random_img(self.drone_imgs)
                bgr_img, bbox = random_insert(bgr_img, drone_img, self.size_range, self.rot_range,
                                              uniform=False, thermal=self.color==2)
                bboxes[i] = bbox

            if self.augmenter is not None:
                bgr_img = self.augmenter.augment_image(bgr_img)

            if self.color == 1:  # if grayscale
                bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_RGB2GRAY)
                bgr_img = np.expand_dims(bgr_img, -1)
            elif self.color == 2:  # if thermal
                bgr_img = bgr_img[..., 0]
                bgr_img = np.expand_dims(bgr_img, -1)

            X[i_batch] = np.divide(bgr_img, 255, dtype='float32')
            if type(self.grid_shape) is list:
                for i, shape in enumerate(self.grid_shape):
                    Y[i][i_batch] = self.bboxes_to_Y(locations, self.img_shape, shape)
            else:
                Y[i_batch] = self.bboxes_to_Y(locations, self.img_shape, self.grid_shape)

        return X, Y

    def on_epoch_end(self):
        pass

def plot_generator_examples(generator, scale_num=0):
    X, Y = generator[0]

    if type(Y) is list:
        bboxes_list = [bboxes_from_Y(Y[scale_num][i], generator.img_shape, 0.5)[0] for i in range(len(Y[scale_num]))]
    else:
        bboxes_list = [bboxes_from_Y(Y[i], generator.img_shape, 0.5)[0] for i in range(len(Y))]

    cols, rows = 3, 2
    fig = plt.figure(figsize = (cols * 5, rows * 5))
    for i in range(rows):
        for j in range(cols):
            idx = i + j * rows
            fig.add_subplot(rows, cols, idx + 1)
            plt.axis('off')
            img = np.squeeze(X[idx])
            plt.imshow(img, cmap='gray')
            for bbox in bboxes_list[idx]:
                x1, y1, x2, y2 = bbox
                plt.gca().add_patch(matplotlib.patches.Rectangle((x1, y1), x2-x1, y2-y1, ec='r', fc='none'))

    plt.tight_layout()


def non_max_suppression(bboxes, confs, threshold):
    '''
    Performs non-max suppression for a single image
    locations shape: (n_coords, 2) or (n_bboxes, 4)
    confs shape: (n_coords, 1)
    Returns coords with most confidence that differ more than dist_thresh from each other
    '''
    best_arr = []  # array of max confidence indexes corresponding to different detected objects
    for i in range(len(bboxes)):
        new_group = True
        for j_idx, j in enumerate(best_arr):
            delta = iou(bboxes[i], bboxes[j])
            if delta > threshold:
                new_group = False
                if confs[i] > confs[j]:
                    best_arr[j_idx] = i
                break

        if new_group:
            best_arr.append(i)

    return bboxes[best_arr], confs[best_arr]


def detect(model, img, threshold, max_detections=100):
    img2 = np.expand_dims(img, 0)
    if len(img2.shape) < 4:
        img2 = np.expand_dims(img2, -1)
    Y_pred = model.predict(img2 / 255)
    if type(Y_pred) is list:
        bboxes, confs = [], []
        for Y in Y_pred:
            out = bboxes_from_Y(Y[0], img.shape, threshold)
            if len(out[0]) > 0:
                locations.append(out[0])
                bboxes.append(out[1])
        if len(bboxes) > 0:
            bboxes = np.concatenate(bboxes)
            confs = np.concatenate(confs)
        else:
            bboxes = np.array(bboxes)
            confs = np.array(confs)
    else:
        bboxes, confs = bboxes_from_Y(Y_pred[0], img.shape, threshold)

    bboxes, confs = non_max_suppression(bboxes, confs, 0.5)
    if len(locations) > max_detections:
        sort_perm = np.argsort(confs)
        bboxes = bboxes[sort_perm][:max_detections]
        confs = confs[sort_perm][:max_detections]

    return bboxes, confs


def display_detections(model, imgs):
    cols, rows = 3, 3
    fig = plt.figure(figsize = (cols * 5, rows * 5))

    for i in range(rows):
        for j in range(cols):
            idx = i + j * cols
            fig.add_subplot(rows, cols, idx + 1)
            plt.axis('off')
            img_num = np.random.choice(len(imgs))
            bboxes, _ = detect(model, imgs[img_num], 0.05)
            plt.imshow(np.squeeze(imgs[img_num]))
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                plt.gca().add_patch(matplotlib.patches.Rectangle((x1, y1), x2-x1, y2-y1, ec='r', fc='none'))

    plt.tight_layout()


def make_detection_csv(model, imgs, paths, output_path, max_detections=100):
    data = pd.DataFrame(columns=('img_name', 'x1', 'y1', 'x2', 'y2', 'p'))
    for i, img in enumerate(imgs):
        bboxes, confs = detect(model, img, 0.05, max_detections=max_detections)
        name = os.path.split(paths[i])[-1]
        for j, bbox in enumerate(bboxes):
            data.loc[len(data)] = (name, bbox[0], bbox[1], bbox[2], bbox[3], confs[j])

    data.to_csv(output_path, index=False)
    return data
