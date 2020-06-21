import tensorflow as tf
import numpy as np
import deep_sort.cfg as cf
import cv2
import time
import os

def track_people():
    print(os.listdir(cf.VIDEO_IMAGES_PATH))
