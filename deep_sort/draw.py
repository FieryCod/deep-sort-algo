import tensorflow as tf
import numpy as np
from operator import itemgetter
import cv2
import time
import random

def iou(img, box1, box2):

    x11, y11 = tuple((box1[0:2] * [img.shape[1], img.shape[0]]).astype(np.int32))
    x12, y12 = tuple((box1[2:4] * [img.shape[1], img.shape[0]]).astype(np.int32))

    x21, y21 = tuple((box2[0:2] * [img.shape[1], img.shape[0]]).astype(np.int32))
    x22, y22 = tuple((box2[2:4] * [img.shape[1], img.shape[0]]).astype(np.int32))

    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou

def setup_tracklets_visualize(img, boxes):
    tracklets = []

    for i in range(boxes.shape[0]):
        right_bottom = (boxes[i, 2:4] * [img.shape[1], img.shape[0]])
        right = right_bottom[0]
        left = (boxes[i, 0:2] * [img.shape[1], img.shape[0]])[0]
        center = tuple(np.array([left + (right - left)/2, right_bottom[1]]).astype(np.int32))
        random_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

        tracklets.append({
            'id': i,
            'box': boxes[i],
            'color': random_color,
            'center_track': center
        })

        cv2.circle(
            img,
            center,
            3,
            random_color,
            3
        )

    return img, tracklets


def tracklet_visualize(img, box, id, color):
    right_bottom = (box[2:4] * [img.shape[1], img.shape[0]])
    right = right_bottom[0]
    left = (box[0:2] * [img.shape[1], img.shape[0]])[0]
    center = tuple(np.array([left + (right - left)/2, right_bottom[1]]).astype(np.int32))

    cv2.circle(
        img,
        center,
        3,
        color,
        3
    )

    return img, {'id': id, 'box': box, 'color': color, 'center_track': center}


def visualize_tracklets(
        img,
        state,
        boxes
):
    last_tracklets = state[-1]
    all_before_last_tracklets = state[:-1]

    iou_current_stats = []
    max_objects = boxes.shape[0]

    for i in range(max_objects):
        current_box = boxes[i]

        for dect in last_tracklets:
            prev_box = dect['box']
            iou_stat = iou(img, prev_box, current_box)

            iou_current_stats.append({'prev_dect': dect, 'current_box': current_box, 'iou_stat': iou_stat})

    sorted_iou_current_stats = sorted(
        filter(lambda t: t['iou_stat'] > 0.5, iou_current_stats),
        key=itemgetter('iou_stat'),
        reverse=True
    )

    max_sorted_iou_current_stats = sorted_iou_current_stats[:7]
    new_tracklets = []

    for stat in max_sorted_iou_current_stats:
        prev_dect = stat['prev_dect']
        id, center_track, box, color = itemgetter('id', 'center_track', 'box', 'color')(prev_dect)
        img1, tracklet = tracklet_visualize(img, stat['current_box'], id, color)
        img2, _ = tracklet_visualize(img1, box, id, color)
        img = img2

        new_tracklets.append(tracklet)

        for before_tracklet_seq in all_before_last_tracklets:
            for before_tracklet in before_tracklet_seq:
                if before_tracklet['id'] == id:
                    img_with_prev_tracklet, _ = tracklet_visualize(img, before_tracklet['box'], id, color)
                    img = img_with_prev_tracklet


    return img, new_tracklets


def track(
        state,
        boxes,
        img
):
    boxes = np.array(boxes)

    if state == None:
        img, tracklets = setup_tracklets_visualize(img, boxes)
        state = [tracklets]
    else:
        img, tracklets = visualize_tracklets(img, state, boxes)
        state.append(tracklets)

    return img, state


def boxes(
        img,
        boxes,
        objectness,
        classes,
        detections,
        class_names,
):
    boxes = np.array(boxes)

    for i in range(detections):
        x1y1 = tuple((boxes[i, 0:2] * [img.shape[1], img.shape[0]]).astype(np.int32))
        x2y2 = tuple((boxes[i, 2:4] * [img.shape[1], img.shape[0]]).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 1)
        img = cv2.putText(
            img,
            '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i]),
            x1y1,
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
            1,
        )

    return img
