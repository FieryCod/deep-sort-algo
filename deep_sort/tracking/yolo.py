import tensorflow as tf
import numpy as np
from deep_sort.debug import bench
import deep_sort.cfg as cf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, \
    Input, ZeroPadding2D, LeakyReLU, UpSampling2D

def non_max_suppression(
        prediction,
        model_size,
        max_output_size,
        max_output_size_per_class,
        iou_threshold,
        confidence_threshold,
):

    (bbox, confs, class_probs) = tf.split(prediction, [4, 1, -1], axis=-1)
    bbox = bbox / model_size[0]
    scores = confs * class_probs
    (boxes, scores, classes, valid_detections) = tf.image.combined_non_max_suppression(
         boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
         scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
         max_output_size_per_class=max_output_size_per_class,
         max_total_size = max_output_size,
         iou_threshold = iou_threshold,
         score_threshold = confidence_threshold,
     )


    return (boxes, scores, classes, valid_detections)

def only_person_boxes(boxes, scores, classes, detections):
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []
    filtered_detections = 0

    for i in range(detections):
        if cf.CLASS_NAMES[int(classes[i])] == cf.PERSON_CLASS_NAME:
            filtered_boxes.append(boxes[i])
            filtered_scores.append(scores[i])
            filtered_classes.append(classes[i])
            filtered_detections += 1

    return (filtered_boxes, filtered_scores, filtered_classes, filtered_detections)

def output_boxes(
        prediction,
        model_size=cf.MODEL_SIZE,
        max_output_size=cf.MAX_OUTPUT_SIZE,
        max_output_size_per_class=cf.MAX_OUTPUT_SIZE_PER_CLASS,
        iou_threshold=cf.IOU_THRESHOLD,
        confidence_threshold=cf.CONFIDENCE_THRESHOLD,
):
    (
        center_x,
        center_y,
        width,
        height,
        confidence,
        classes,
    ) = tf.split(
        prediction, [
            1,
            1,
            1,
            1,
            1,
            -1,
        ],
        axis=-1
    )

    top_left_x = center_x - width / 2.0
    top_left_y = center_y - height / 2.0
    bottom_right_x = center_x + width / 2.0
    bottom_right_y = center_y + height / 2.0

    prediction = tf.concat(
        [
            top_left_x,
            top_left_y,
            bottom_right_x,
            bottom_right_y,
            confidence,
            classes,
        ],
        axis=-1
    )

    boxes, scores, classes, detections = non_max_suppression(
        prediction,
        model_size,
        max_output_size,
        max_output_size_per_class,
        iou_threshold,
        confidence_threshold,
    )

    person_boxes = only_person_boxes(boxes[0], scores[0], classes[0], detections[0])

    return person_boxes


def resize_image(inputs, modelsize):
    inputs = tf.image.resize(inputs, modelsize)

    return inputs


def init(blocks, model_size, num_classes):
    outputs = {}
    output_filters = []
    filters = []
    out_pred = []
    scale = 0

    inputs = input_image = Input(shape=model_size)
    inputs = inputs / 255.0

    for i, block in enumerate(blocks[1:]):
        # If it is a convolutional layer
        if (block["type"] == "convolutional"):

            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])

            if strides > 1:
                inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)

            inputs = Conv2D(
                filters,
                kernel_size,
                strides=strides,
                padding='valid' if strides > 1 else 'same',
                name='conv_' + str(i),
                use_bias=False if ("batch_normalize" in block) else True
            )(inputs)

            if "batch_normalize" in block:
                inputs = BatchNormalization(name='bnorm_' + str(i))(inputs)

                if activation == "leaky":
                    inputs = LeakyReLU(alpha=0.1, name='leaky_' + str(i))(inputs)

        elif (block["type"] == "upsample"):
            stride = int(block["stride"])
            inputs = UpSampling2D(stride)(inputs)

        # If it is a route layer
        elif (block["type"] == "route"):
            block["layers"] = block["layers"].split(',')
            start = int(block["layers"][0])

            if len(block["layers"]) > 1:
                end = int(block["layers"][1]) - i
                filters = output_filters[i + start] + output_filters[end]  # Index negatif :end - index
                inputs = tf.concat(
                    [outputs[i + start], outputs[i + end]], axis=-1)
            else:
                filters = output_filters[i + start]
                inputs = outputs[i + start]

        elif block["type"] == "shortcut":
            from_ = int(block["from"])
            inputs = outputs[i - 1] + outputs[i + from_]

        # Yolo detection layer
        elif block["type"] == "yolo":

            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]
            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            n_anchors = len(anchors)

            out_shape = inputs.get_shape().as_list()

            inputs = tf.reshape(inputs, [-1, n_anchors * out_shape[1] * out_shape[2], 5 + num_classes])

            box_centers = inputs[:, :, 0:2]
            box_shapes = inputs[:, :, 2:4]
            confidence = inputs[:, :, 4:5]
            classes = inputs[:, :, 5:num_classes + 5]

            box_centers = tf.sigmoid(box_centers)
            confidence = tf.sigmoid(confidence)
            classes = tf.sigmoid(classes)

            anchors = tf.tile(anchors, [out_shape[1] * out_shape[2], 1])
            box_shapes = tf.exp(box_shapes) * tf.cast(
                anchors, dtype=tf.float32)

            x = tf.range(out_shape[1], dtype=tf.float32)
            y = tf.range(out_shape[2], dtype=tf.float32)

            cx, cy = tf.meshgrid(x, y)
            cx = tf.reshape(cx, (-1, 1))
            cy = tf.reshape(cy, (-1, 1))
            cxy = tf.concat([cx, cy], axis=-1)
            cxy = tf.tile(cxy, [1, n_anchors])
            cxy = tf.reshape(cxy, [1, -1, 2])

            strides = (input_image.shape[1] // out_shape[1],
                       input_image.shape[2] // out_shape[2])
            box_centers = (box_centers + cxy) * strides

            prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

            if scale:
                out_pred = tf.concat([out_pred, prediction], axis=1)
            else:
                out_pred = prediction
                scale = 1

        outputs[i] = inputs
        output_filters.append(filters)

    model = Model(input_image, out_pred)

    return model


def full_model():
    partial_model = init(cf.BLOCKS, cf.MODEL_SIZE, cf.CLASSES_USED)
    partial_model.load_weights(cf.MODEL_FILE)

    return partial_model


@bench
def predict(model, image):
    image = resize_image(tf.expand_dims(np.array(image), 0), (cf.MODEL_SIZE[0], cf.MODEL_SIZE[1]))

    return model.predict(image)
