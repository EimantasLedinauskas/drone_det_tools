'''
Purpose: Analyse predictions using
        detections.csv   --- detections output file
                             with format: img_name, x1, y1, x2, y2, p

        annotations.csv  --- annotations RetinaNet style:
                             format: img_name, x1, y1, x2, y2, class_name

        Assuming maximum one drone per image !
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from tqdm import tqdm
import cProfile


# ============================================================================ #
#                                MISC FUNCTIONS                                #
# ============================================================================ #

def iou(box1, box2):
    '''
    Box format is: x1,y1,x2,y2 :thinking:
    '''

    # Intersection
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[0] + box1[2], box2[0] + box2[2])
    yi2 = np.minimum(box1[1] + box1[3], box2[1] + box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # Union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def coords_to_bbox(coords, bbox):
    '''
    Translates bbox so that its center would be at coords. Used to convert
    predicted center coordinates to a bounding box for simple comparison.
    '''

    # Required translations
    delta_x = coords[0] - 0.5 * (bbox[0] + bbox[2])
    delta_y = coords[1] - 0.5 * (bbox[1] + bbox[3])

    return (bbox[0] + delta_x, bbox[1] + delta_y,
            bbox[2] + delta_x, bbox[3] + delta_y)


# ============================================================================ #
#                                  READING CSV                                 #
# ============================================================================ #

def get_gt_annotations(dataset_name):
    '''
    Read gt annotations by dataset name
    '''
    return pd.read_csv(
        os.path.join('true_annotations', dataset_name+'__annotations.csv'),
        names=['img_name', 'x1', 'y1', 'x2', 'y2', 'class']
    )

# true_annotations_df = get_gt_annotations('single_youtube_drone')
# pred_annotations_df = pd.read_csv(os.path.join('predictions', names[0], datasets[0], 'detections.csv'))


# -------------------- Concerting dataframe to dictionary -------------------- #

def get_detection_dictionaries(true_annotations_df, pred_annotations_df):
    '''
    Purpose: convert pandas dataframes to dicts in required format

    Input
        true_annotations_df --- true annotation dataframe
        pred_annotations_df --- predicted annotations dataframe

    Returns:
        true_annotations --- true_annotations[img_name] = bbox
        pred_annotations --- predicted annotations dictionary pred_annotations[img_name] = [(bbox, p), ...]

    #TODO: Make a faster conversion using pandas builtin methods
    '''

    img_names = list(set(list(pred_annotations_df.img_name)))
    N_detections = len(img_names)

    true_annotations = {}
    pred_annotations = defaultdict(list)

    n_skipped = 0

    # iterating over images used to test the network
    for img_name in tqdm(img_names):

        try:
            row_gt = true_annotations_df.loc[true_annotations_df['img_name'] == img_name].iloc[0]
        except IndexError:
            n_skipped += 1
            continue

        bbox_true = tuple(row_gt[['x1', 'y1', 'x2', 'y2']])

        true_annotations[img_name] = bbox_true

        for index, row in pred_annotations_df.loc[pred_annotations_df['img_name'] == img_name].iterrows():
            # read prediction from dataframe row
            bbox_pred = tuple(row[['x1', 'y1', 'x2', 'y2']])
            p_pred = row[['p']][0]

            pred_annotations[img_name].append((bbox_pred, p_pred))

    print('Nr. of skipped imgs:', n_skipped)
    return true_annotations, pred_annotations


# ============================================================================ #
#                                COUNTING STATS                                #
# ============================================================================ #


# -------------------------- single image detections ------------------------- #
def process_image_detections(img_name, pred_annotations, true_annotations, only_coords=False):
    '''
    Return bbox overlaps and probs for one image.

    img_name --- image name
    pred_annotations --- predicted annotations dictionary pred_annotations[img_name] = [(bbox, p), ...]
    true_annotations --- true_annotations[img_name] = bbox

    bbox format is (x1, y1, x2, y2)
    '''

    overlaps = []
    probs = []

    bbox_true = true_annotations[img_name]

    # check if there is a drone at current frame
    if np.isnan(bbox_true[0]):
        n_drones_true = 0
    else:
        n_drones_true = 1  # assuming only one drone

    # iterating over predictions
    for bbox_pred, p_pred in pred_annotations[img_name]:
        probs.append(p_pred)
        if n_drones_true == 0:
            # compute iou
            overlaps.append(0)  # zero iou if there is no drone
        else:
            if only_coords:
                bbox_pred = coords_to_bbox(bbox_pred[:2], bbox_true)

            overlaps.append(iou(bbox_pred, bbox_true))

    return np.array(overlaps), np.array(probs), n_drones_true


# ------------------------------- whole dataset ------------------------------ #
def get_detection_stats(pred_annotations,
                        true_annotations,
                        p_thresh,
                        iou_thresh,
                        only_coords=False):
    '''
    Purpose: count the number of
        TN, TP, FN, FP
        given single p_thresh value

    pred_annotations  --- dict of predicted annotations
                            pred_annotations['img_name'] = [(bbox, p), ... ]

    true_annotations  --- dict of true annotations (assuming one drone only)
                            true_annotations['img_name'] = bbox

    p_thresh          ---  detection treshold
    iou_thresh        ---  iou treshold

    '''
    detections = defaultdict(list)

    # statistics
    TN = 0
    TP = 0
    FP = 0
    FN = 0

    img_names = list(pred_annotations.keys())
    N_detections = len(img_names)

    for img_name in (img_names):
        # load all overlaps and probs for an image
        overlaps, probs, n_drones_true = process_image_detections(img_name,
                                                                  pred_annotations,
                                                                  true_annotations,
                                                                  only_coords=only_coords)

        good_overlap = overlaps > iou_thresh
        confident = probs > p_thresh

        # ---------------------------- detections logic ---------------------------

        # true positive ---  one (in future may be many) per image
        if np.any(good_overlap * confident):
            # is there any confident detection with good overlap
            # assuming that there can only be one object
            TP += 1

        # true negative --- one per image ---  there are no confident detections and no drones
        if (not np.any(confident)) and (n_drones_true == 0):
            TN += 1

        # false negative --- one per image --- there is a drone but no correct and confident bbox
        if (n_drones_true > 0) and not np.any(confident * good_overlap):
            FN += 1

        # iterate over all detections at single image
        for n in range(len(confident)):
            # false positive --- can be many
            if confident[n] and (not good_overlap[n]):
                # confident prediction with too small iou
                FP += 1

    return TN, TP, FN, FP



def compute_detection_stats_vs_p_thresh(pred_annotations,
                                        true_annotations,
                                        p_min=0.05,
                                        p_max=0.9,
                                        only_coords=False):
    '''
    Purpose: compute detection statistics by varying p_threshold

    Input:
        pred_annotations  --- dict of predicted annotations
                                pred_annotations['img_name'] = [(bbox, p), ... ]

        true_annotations  --- dict of true annotations (assuming one drone only)
                                true_annotations['img_name'] = bbox

    Output:
        TN, TP, FN, FP, P
    '''

    precision = []
    recall = []

    TN, TP, FN, FP = [], [], [], []

    P = np.linspace(p_min, p_max, 50)
    for p in tqdm(P):
        tn, tp, fn, fp = get_detection_stats(pred_annotations,
                                             true_annotations,
                                             p_thresh=p,
                                             iou_thresh=0.5,
                                             only_coords=only_coords)
        TN.append(tn)
        TP.append(tp)
        FN.append(fn)
        FP.append(fp)

    return tuple( map(np.array, (TN, TP, FN, FP, P)) )

# ============================================================================ #
#                                     PLOTS                                    #
# ============================================================================ #



def plot_detection_analysis(TN, TP, FN, FP, P):


    precision = (TP / (TP + FP + 1e-16))
    recall = (TP / (TP + FN + 1e-16))

    plt.figure(figsize=(7, 4))
    plt.plot(recall, precision)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    plt.plot(P, TN)
    plt.xlabel('p_thresh')
    plt.title('True Negative')

    plt.subplot(2, 2, 2)
    plt.plot(P, TP)
    plt.xlabel('p_thresh')
    plt.title('True Positive')

    plt.subplot(2, 2, 3)
    plt.plot(P, FN)
    plt.xlabel('p_thresh')
    plt.title('False Negative')

    plt.subplot(2, 2, 4)
    plt.plot(P, FP)
    plt.xlabel('p_thresh')
    plt.title('False Positive')
    plt.subplots_adjust(hspace=0.5)
    plt.show()




# ============================================================================ #
#                                 USAGE EXAMPLE                                #
# ============================================================================ #

if __name__ == 'main':

    # read annotations: true and predicted
    pred_annotations_df = pd.read_csv('detections.csv')
    true_annotations_df = pd.read_csv('annotations.csv')

    # Convert dataframes  to dictionaries for speed
    true_annotations, pred_annotations = get_detection_dictionaries(true_annotations_df, pred_annotations_df)

    # Compute the relevant detection statistics
    TN, TP, FN, FP, P = compute_detection_stats_vs_p_thresh(pred_annotations, true_annotations)

    plot_detection_analysis(TN, TP, FN, FP, P)
