import deep_sort.tracking.yolo as yolo
from natsort import natsorted
import deep_sort.draw as draw
import tensorflow as tf
import numpy as np
import deep_sort.cfg as cf
import operator
import cv2
import time
import os

def track_people():
    input_imgs_paths = list(filter(lambda p : operator.contains(p, '.jpg'), os.listdir(cf.VIDEO_IMAGES_PATH)))
    model = yolo.full_model()

    for _, img_path in enumerate(natsorted(input_imgs_paths)):
        print(img_path)
        img = cv2.imread(os.path.join(cf.VIDEO_IMAGES_PATH, img_path))
        prediction = yolo.predict(model, img)
        boxes, scores, classes, nums = yolo.output_boxes(prediction)
        img_with_boxes = draw.boxes(img, boxes, scores, classes, nums, cf.CLASS_NAMES)

        cv2.imwrite(os.path.join(cf.VIDEO_IMAGES_OUTPUT_PATH, img_path), img_with_boxes)


track_people()
