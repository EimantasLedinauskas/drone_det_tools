import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_img_paths(img_dir):
    paths = []
    for i, fname in enumerate(os.listdir(img_dir)):
        if fname[-3:] in ['png', 'jpg', 'JPG', 'PNG', 'JPEG', 'jpeg']:
            paths.append(os.path.join(img_dir, fname))
    return paths


def load_imgs(paths, grayscale=False):
    imgs = np.empty((len(paths), *IMG_SHAPE), dtype='uint8') if grayscale else \
           np.empty((len(paths), *IMG_SHAPE, 3), dtype='uint8')
    for i, path in enumerate(paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        imgs[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if grayscale else \
              cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(imgs)


def load_imgs_dir(img_dir, colors=cv2.IMREAD_COLOR):
    imgs = []
    for fname in os.listdir(img_dir):
        if fname[-3:] in ['png', 'jpg', 'JPG', 'PNG', 'JPEG', 'jpeg']:
            img = cv2.imread(os.path.join(img_dir, fname), colors)
            if colors == cv2.IMREAD_UNCHANGED:
                imgs.append(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
            elif colors == cv2.IMREAD_COLOR:
                imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.array(imgs)


def plot_history_item(name, hists, ax, log=False, plot_val=True):
    y = [x for hist in hists for x in hist.history[name]]
    ax.plot(y, label="Train " + name)
    if plot_val:
        y = [x for hist in hists for x in hist.history["val_" + name]]
        ax.plot(y, label="Validation " + name)
    if log:
        ax.set_yscale('log')
    ax.grid()
    ax.legend()
