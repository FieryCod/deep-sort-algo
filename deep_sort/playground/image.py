import cv2
import sys
import numpy as np
import tensorflow as tf
import deep_sort.draw as draw
import deep_sort.tracking.yolo as yolo
import deep_sort.cfg as cf

def main(args):
    model = yolo.full_model()
    image = cv2.imread(args[0])
    prediction = yolo.predict(model, image)
    boxes, scores, classes, nums = yolo.output_boxes(prediction)
    img = draw.boxes(image, boxes, scores, classes, nums, cf.CLASS_NAMES)

    cv2.imshow('Image detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1:])
