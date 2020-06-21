import tensorflow as tf
from tensorflow.python.client import device_lib


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()

    return [x.name for x in local_device_protos]


def reconfigure_tensorflow(gpu_devices):
    assert len(gpu_devices) > 0, "Not enough GPU hardware devices available. Exiting.."
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def parse_cfg(cfgfile):
    with open(cfgfile, 'r') as file:
        lines = [
            line.rstrip('\n') for line in file
            if line != '\n' and line[0] != '#'
        ]
    holder = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            line = 'type=' + line[1:-1].rstrip()
            if len(holder) != 0:
                blocks.append(holder)
                holder = {}
        key, value = line.split("=")
        holder[key.rstrip()] = value.lstrip()

    blocks.append(holder)

    return blocks


WEIGHTS_FILE = "resources/cfg/weights/yolov3.weights"
YOLO_CFG_FILE = "resources/cfg/yolov3.cfg"
MODEL_FILE = "resources/model/yolo.tf"
COCO_NAMES = "resources/cfg/coco.names"
MODEL_SIZE = (416, 416, 3)
CLASSES_USED = 80 # Corresponds to coco.names # 80
GPU_DEVICES = tf.config.experimental.list_physical_devices('GPU')
MAX_OUTPUT_SIZE = 40
MAX_OUTPUT_SIZE_PER_CLASS = 20
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
BLOCKS = parse_cfg(YOLO_CFG_FILE)
OUTPUT_PATH = "output"

# Further configure & read data
reconfigure_tensorflow(GPU_DEVICES)
