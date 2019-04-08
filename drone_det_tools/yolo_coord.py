from drone_det_tools.fake_imgs import random_insert
from drone_det_tools.misc import load_imgs

import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import Sequence


def convert_to_Y(coords, img_shape, grid_shape):
    Y = np.zeros((*grid_shape, 3), dtype='float32')
    grid_l_x = img_shape[1] / grid_shape[1]
    grid_l_y = img_shape[0] / grid_shape[0]
    for coord in coords:
        x, y = coord
        grid_j = int(x / grid_l_x)
        grid_i = int(y / grid_l_y)
        Y[grid_i, grid_j, 0] = 1.0
        Y[grid_i, grid_j, 1] = x / grid_l_x - grid_j
        Y[grid_i, grid_j, 2] = y / grid_l_y - grid_i

    return Y


def convert_from_Y(Y, img_shape, threshold):
    coords = []  # x,y pairs
    confs = []  # confidences
    grid_shape = Y.shape[0:2]
    grid_l_x = img_shape[1] / grid_shape[1]
    grid_l_y = img_shape[0] / grid_shape[0]

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if Y[i, j, 0] > threshold:
                x = (Y[i, j, 1] + j) * grid_l_x
                y = (Y[i, j, 2] + i) * grid_l_y
                coords.append((x, y))
                confs.append(Y[i, j, 0])

    return np.array(coords), np.array(confs)


class CoordFakeGenerator(Sequence):

    def __init__(self, bgr_paths, drone_paths, batch_size, batches_per_epoch,
                 size_range, rot_range, img_shape, grid_shape,
                 bird_paths=None, bgr_augmenter=None, augmenter=None, grayscale=True, thermal=False):

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
        self.grayscale = grayscale
        self.thermal = thermal

        self.on_epoch_end()

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):

        n_channels = 1 if self.grayscale or self.thermal else 3
        X = np.empty((self.batch_size, *self.img_shape, n_channels), dtype='float32')
        if type(self.grid_shape) is list:
            Y = [np.empty((self.batch_size, *shape, 3), dtype='float32') for shape in self.grid_shape]
        else:
            Y = np.empty((self.batch_size, self.grid_shape, 3), dtype='float32')

        for i_batch in range(self.batch_size):

            bgr_idx = np.random.choice(len(self.bgr_imgs))
            bgr_img = self.bgr_imgs[bgr_idx]

            if self.bgr_augmenter is not None:
                bgr_img = self.bgr_augmenter.augment_image(bgr_img)

            if self.bird_imgs is not None:
                n_birds = np.random.choice(4)
                for i in range(n_birds):
                    bird_idx = np.random.choice(len(self.bird_imgs))
                    bird_img = self.bird_imgs[bird_idx]
                    bgr_img, _, _ = random_insert(bgr_img, bird_img, self.size_range, self.rot_range, uniform=False, thermal=self.thermal)

            n_drones = np.random.choice(range(1,4))
            coords = np.empty((n_drones, 2))
            for i in range(n_drones):
                drone_idx = np.random.choice(len(self.drone_imgs))
                drone_img = self.drone_imgs[drone_idx]
                bgr_img, x, y = random_insert(bgr_img, drone_img, self.size_range, self.rot_range, uniform=False, thermal=self.thermal)
                coords[i] = x, y

            if self.augmenter is not None:
                bgr_img = self.augmenter.augment_image(bgr_img)

            if self.grayscale and not self.thermal:
                bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_RGB2GRAY)
                bgr_img = np.expand_dims(bgr_img, -1)

            if self.thermal:
                bgr_img = bgr_img[..., 0]
                bgr_img = np.expand_dims(bgr_img, -1)

            X[i_batch] = np.divide(bgr_img, 255, dtype='float32')
            if type(self.grid_shape) is list:
                for i, shape in enumerate(self.grid_shape):
                    Y[i][i_batch] = convert_to_Y(coords, self.img_shape, shape)
            else:
                Y[i_batch] = convert_to_Y(coords, self.img_shape, self.grid_shape)

        return X, Y

    def on_epoch_end(self):
        pass

def plot_generator_examples(generator, scale_num=0):
    X, Y = generator[0]
    coords_list = [convert_from_Y(Y[scale_num][i], generator.img_shape, 0.5)[0] for i in range(len(Y[scale_num]))]
    cols, rows = 3, 2
    fig = plt.figure(figsize = (cols * 5, rows * 5))
    for i in range(rows):
        for j in range(cols):
            idx = i + j * rows
            fig.add_subplot(rows, cols, idx + 1)
            plt.axis('off')
            img = np.squeeze(X[idx])
            plt.imshow(img, cmap='gray')
            for coord in coords_list[idx]:
                x, y = coord
                circle = plt.Circle((x, y), 30, color='g', fill=False, linewidth=3)
                plt.gca().add_artist(circle)

    plt.tight_layout()


def non_max_suppression(coords, confs, dist_thresh):
    '''
    Performs non-max suppression for a single image
    coords shape: (n_coords, 2)
    confs shape: (n_coords, 1)
    Returns coords with most confidence that differ more than dist_thresh from each other
    '''
    best_arr = []  # array of max confidence indexes corresponding to different detected objects
    for i in range(len(coords)):
        new_group = True
        for j_idx, j in enumerate(best_arr):
            dist = np.sqrt(np.sum(np.square(coords[i] - coords[j])))
            if dist < dist_thresh:
                new_group = False
                if confs[i] > confs[j]:
                    best_arr[j_idx] = i
                break

        if new_group:
            best_arr.append(i)

    return coords[best_arr], confs[best_arr]


def detect(model, img, threshold):
    img2 = np.expand_dims(img, 0)
    img2 = np.expand_dims(img2, -1)
    Y_pred = model.predict(img2 / 255)
    if type(Y_pred) is list:
        coords, confs = [], []
        for Y in Y_pred:
            out = convert_from_Y(Y[0], img.shape, threshold)
            if len(out[0]) > 0:
                coords.append(out[0])
                confs.append(out[1])
        if len(coords) > 0:
            coords = np.concatenate(coords)
            confs = np.concatenate(confs)
    else:
       coords, confs = convert_from_Y(Y_pred[0], img.shape, threshold)

    coords, confs = non_max_suppression(coords, confs, 0.1 * img.shape[0])
    return coords, confs


def display_detections(model, imgs):
    cols = 3
    rows = 3
    fig=plt.figure(figsize = (cols * 5, rows * 5))

    for i in range(rows):
        for j in range(cols):
            idx = i + j * cols
            fig.add_subplot(rows, cols, idx + 1)
            plt.axis('off')
            img_num = np.random.choice(len(imgs))
            coords, _ = detect(model, imgs[img_num], 0.05)
            plt.imshow(imgs[img_num])
            for coord in coords:
                x, y = coord
                circle = plt.Circle((x, y), 30, color='r', fill=False, linewidth=3)
                plt.gca().add_artist(circle)

    plt.tight_layout()


def make_detection_csv(model, imgs, paths, output_path):
    data = pd.DataFrame(columns=('name', 'x1', 'y1', 'x2', 'y2', 'p'))
    for i, img in enumerate(imgs):
        coords, confs = detect(model, img, 0.05)
        name = os.path.split(paths[i])[-1]
        for j, coord in enumerate(coords):
            data.loc[len(data)] = (name, coord[0], coord[1], np.NaN, np.NaN, confs[j])
    data.to_csv(output_path, index=False)
    return data
