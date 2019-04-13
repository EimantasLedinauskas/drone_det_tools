import numpy as np
import cv2


def find_i_min_i_max(arr, axis, threshold):
    sums = np.sum(arr, axis)
    i_min = 0
    while sums[i_min] < threshold:
        i_min += 1

    i_max = len(sums) - 1
    while sums[i_max] < threshold:
        i_max -= 1

    return i_min, i_max


def trim_image(img, threshold=5):
    alpha_ch = img[:, :, 3]
    i_min, i_max = find_i_min_i_max(alpha_ch, 0, threshold)
    j_min, j_max = find_i_min_i_max(alpha_ch, 1, threshold)
    return img[j_min:j_max, i_min:i_max]


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
    mask = np.stack((subimg[:, :, 3] / 255, subimg[:, :, 3] / 255, subimg[:, :, 3] / 255), axis=2)
    y1 = row
    y2 = row+subimg.shape[0]
    x1 = col
    x2 = col+subimg.shape[1]
    result[y1:y2, x1:x2] = np.uint8( result[y1:y2, x1:x2] * (1 - mask) )
    result[y1:y2, x1:x2] += np.uint8(mask * subimg[:, :, :3])
    blured = cv2.GaussianBlur(result[y1:y2, x1:x2], (3,3), 0)
    result[y1:y2, x1:x2] = np.uint8( result[y1:y2, x1:x2] * (1 - mask) )
    result[y1:y2, x1:x2] += np.uint8(mask * blured)
    return result


def random_insert(img, subimg, size_range, angle_range, coords_only=False,
                  uniform=True, thermal=False):
    min_size, max_size = size_range
    min_angle, max_angle = angle_range

    if uniform:
        size = np.random.uniform(min_size, max_size)
    else:
        size = np.exp(np.random.uniform(np.log(min_size), np.log(max_size)))

    size = size * min(img.shape[0], img.shape[1])
    scale = size / max(subimg.shape[0], subimg.shape[1])
    subimg_resc = cv2.resize(subimg, (int(subimg.shape[1] * scale), int(subimg.shape[0] * scale)))

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

    if np.random.rand() < 0.5:
        subimg_resc = cv2.flip(subimg_resc, 1);

    row = np.random.randint(img.shape[0] - subimg_resc.shape[0])
    col = np.random.randint(img.shape[1] - subimg_resc.shape[1])

    if coords_only:
        x = int(round(col + subimg_resc.shape[1] / 2))
        y = int(round(row + subimg_resc.shape[0] / 2))
        return insert_subimg(img, subimg_resc, row, col), (x, y)
    else:
        x1 = int(col)
        y1 = int(row)
        x2 = int(col + subimg_resc.shape[1])
        y2 = int(row + subimg_resc.shape[0])
        return insert_subimg(img, subimg_resc, row, col), (x1, y1, x2, y2)
