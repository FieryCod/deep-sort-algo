import tensorflow as tf
import numpy as np
import cv2
import time

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
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(
            img,
            '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i]),
            x1y1,
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
            2,
        )

    return img
