import cv2
import sys
import numpy as np
import tensorflow as tf
from deep_sort.impl.draw import load_class_names, output_boxes, draw_outputs, resize_image
from deep_sort.impl.yolo import YOLOv3Net
import deep_sort.cfg as cf

def main(args):
    model = YOLOv3Net(cf.BLOCKS, cf.MODEL_SIZE, cf.CLASSES_USED)
    model.load_weights(cf.MODEL_FILE)

    class_names = load_class_names(cf.COCO_NAMES)
    image = cv2.imread(args[0])
    image = np.array(image)
    image = tf.expand_dims(image, 0)
    resized_frame = resize_image(image, (cf.MODEL_SIZE[0], cf.MODEL_SIZE[1]))
    pred = model.predict(resized_frame)
    boxes, scores, classes, nums = output_boxes( \
        pred, cf.MODEL_SIZE,
        max_output_size=cf.MAX_OUTPUT_SIZE,
        max_output_size_per_class=cf.MAX_OUTPUT_SIZE_PER_CLASS,
        iou_threshold=cf.IOU_THRESHOLD,
        confidence_threshold=cf.CONFIDENCE_THRESHOLD
    )
    image = np.squeeze(image)
    img = draw_outputs(image, boxes, scores, classes, nums, class_names)

    cv2.imshow('Image detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1:])
