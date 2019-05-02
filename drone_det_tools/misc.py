import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
import keras.backend as K

def read_img_paths(img_dir):
    paths = []
    for i, fname in enumerate(os.listdir(img_dir)):
        if fname[-3:] in ['png', 'jpg', 'JPG', 'PNG'] or fname[-4:] in ['jpeg', 'JPEG']:
            paths.append(os.path.join(img_dir, fname))
    return np.array(paths)


def load_imgs(paths, img_shape=None, grayscale=False, alpha=False):
    if alpha:
        n_channels = 4
        colors = cv2.IMREAD_UNCHANGED
        conversion = cv2.COLOR_BGRA2RGBA
    elif grayscale:
        n_channels = 1
        colors = cv2.IMREAD_GRAYSCALE
    else:
        n_channels = 3
        colors = cv2.IMREAD_COLOR
        conversion = cv2.COLOR_BGR2RGB

    imgs = [] if img_shape is None else \
           np.empty((len(paths), *img_shape, n_channels), dtype='uint8')

    for i, path in enumerate(paths):
        img = cv2.imread(path, colors)
        if grayscale:
            img = np.expand_dims(img, -1)
        else:
            img = cv2.cvtColor(img, conversion)
        if img_shape is not None:
            imgs[i] = img if img.shape[:2] == img_shape else cv2.resize(img, img_shape[::-1])
        else:
            imgs.append(img)

    if img_shape is None:
        return np.array(imgs)
    else:
        return imgs


def load_imgs_dir(img_dir, img_shape=None, grayscale=False, alpha=False):
    paths = read_img_paths(img_dir)
    return load_imgs(paths, img_shape, grayscale, alpha)


def plot_history_item(name, hists, ax, log=False, plot_val=True, use_ends_with=False):
    if use_ends_with:
        names = []
        for key in hists[0].history.keys():
            if key.endswith(name) and not key.startswith('val') :
                names.append(key)
        color_vals = np.linspace(0.4, 1.0, len(names))
    else:
        names = [name]
        color_vals = [1.0]


    for key, color_val in zip(names, color_vals):
        y = [x for hist in hists for x in hist.history[key]]
        ax.plot(y, label="Train " + key, color=(0, 0, color_val))
        if plot_val:
            y = [x for hist in hists for x in hist.history["val_" + key]]
            ax.plot(y, label="Validation " + key, color=(color_val, 0, 0))

    if log:
        ax.set_yscale('log')

    ax.grid()
    ax.legend()


def short_summary(model):
    print('Input shapes:', end='')
    for input in model.inputs:
        print('', input.shape, end='')
    print('.')

    print('Output shapes:', end='')
    for output in model.outputs:
        print('', output.shape, end='')
    print('.')

    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
