import numpy as np
import cv2


def trim_image(img):
    x, y, w, h = cv2.boundingRect(img[:, :, 3])
    return img[y:y+h, x:x+w]


def rotate_img(img, angle):
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    abs_cos = abs(M[0,0])
    abs_sin = abs(M[0,1])
    cols_rot = int(rows * abs_sin + cols * abs_cos)
    rows_rot = int(rows * abs_cos + cols * abs_sin)
    M[0, 2] += (cols_rot - cols) / 2
    M[1, 2] += (rows_rot - rows) / 2
    result = cv2.warpAffine(img, M, (cols_rot, rows_rot))
    return trim_image(result)


def insert_subimg(img, subimg, row, col):
    result = np.copy(img)
    mask = subimg[:, :, 3] / 255
    mask = np.stack((mask, mask, mask), axis=2)
    x1, y1 = col, row
    x2, y2 = col + subimg.shape[1], row + subimg.shape[0]
    result[y1:y2, x1:x2] = np.uint8( result[y1:y2, x1:x2] * (1 - mask) )
    result[y1:y2, x1:x2] += np.uint8(mask * subimg[:, :, :3])
    blured = cv2.GaussianBlur(result[y1:y2, x1:x2], (3,3), 0)
    result[y1:y2, x1:x2] = np.uint8( result[y1:y2, x1:x2] * (1 - mask) )
    result[y1:y2, x1:x2] += np.uint8(mask * blured)
    return result


def smoothize_alpha(img):
    mask = img[:, :, 3]
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    img[:, :, 3] = mask
    return img


def random_insert(img, subimg, size_range, angle_range, uniform=True, thermal=False, feathering=False,
                 only_coords=False):
    min_size, max_size = size_range
    min_angle, max_angle = angle_range

    if feathering:
        subimg = smoothize_alpha(subimg)

    if uniform:
        size = np.random.uniform(min_size, max_size)
    else:
        size = np.exp(np.random.uniform(np.log(min_size), np.log(max_size)))

    size = size * min(img.shape[0], img.shape[1])
    scale = size / max(subimg.shape[0], subimg.shape[1])
    new_w = int(subimg.shape[1] * scale)
    new_h = int(subimg.shape[0] * scale)
    if new_h == 0 or new_w == 0:
        if only_coords:
            return img, (0, 0)
        return img, (0, 0, 0, 0)
    subimg_resc = cv2.resize(subimg, (new_w, new_h))

    if thermal:
        subimg_resc[..., :3] = 255 - subimg_resc[..., :3]
        subimg_resc[..., :3] = np.uint8(np.clip(subimg_resc[..., :3] * np.random.uniform(1.0, 1.5), 0, 255))
    else:
        # shuffle color channels
        perm = np.random.permutation(3)
        perm = np.append(perm, 3)
        subimg_resc = subimg_resc[:, :, perm]

    angle = np.random.uniform(min_angle, max_angle)
    subimg_resc = rotate_img(subimg_resc, angle)

    if subimg_resc.shape[0] == 0 or subimg_resc.shape[1] == 0:
        if only_coords:
            return img, (0, 0)
        return img, (0, 0, 0, 0)

    if np.random.rand() < 0.5:
        subimg_resc = cv2.flip(subimg_resc, 1);

    row = np.random.randint(img.shape[0] - subimg_resc.shape[0])
    col = np.random.randint(img.shape[1] - subimg_resc.shape[1])

    if only_coords:
        x = int(round(col + subimg_resc.shape[1] / 2))
        y = int(round(row + subimg_resc.shape[0] / 2))
        return insert_subimg(img, subimg_resc, row, col), (x, y)

    x1 = int(col)
    y1 = int(row)
    x2 = int(col + subimg_resc.shape[1])
    y2 = int(row + subimg_resc.shape[0])
    return insert_subimg(img, subimg_resc, row, col), (x1, y1, x2, y2)


def random_fragment(img, width, height):
    width = int(width * img.shape[1])
    height = int(height * img.shape[0])
    x = np.random.randint(img.shape[1] - width)
    y = np.random.randint(img.shape[0] - height)
    return img[y:y+height, x:x+width]


def random_fragmentized_insert(img, subimg, n_fragments, fragment_size_range, size_range,
                               angle_range, uniform=True, thermal=False):
    subimg_trimed = trim_image(subimg)
    for _ in range(n_fragments):
        width = np.random.uniform(fragment_size_range[0], fragment_size_range[1])
        height = np.random.uniform(fragment_size_range[0], fragment_size_range[1])
        fragment = random_fragment(subimg_trimed, width, height)
        if np.sum(fragment[..., 3]) < 5:
            continue
        img, _ = random_insert(img, fragment, size_range, angle_range, uniform=uniform,
                               thermal=thermal, feathering=True, only_coords=True)
    return img
