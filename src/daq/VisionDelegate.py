from math import pi
from threading import Lock
from typing import Dict, List
import tflite_runtime.interpreter as tflite
from utils.cv_utils import detect_objects
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.cv_utils import colors

CP_THRESHOLD = pi / 12
class VisionDelegate:
    debug_initialized = False
    _unclassifiedDetections: List[Dict[str, any]] = None
    _threadLock: Lock = None
    img: Image
    
    def __init__(self, model_path, numthreads=32, threshold=0.5):
        self._threshold = threshold
        self._interpreter = tflite.Interpreter(model_path=model_path, num_threads=numthreads)
        self._interpreter.allocate_tensors()
        self._threadLock = Lock()

        input_shape = self._interpreter.get_input_details()[0]['shape']
        self._height = input_shape[1]
        self._width = input_shape[2]
        self._unclassifiedDetections = []
        self._draw_init = False
        self.img = None

    def draw(self):
        if self.img is None:
            return
        if not self._draw_init:
            (self.fig, self.ax) = plt.subplots()
            plt.ion()
            plt.show()
            self._draw_init = True

        plt.cla()
        self.ax.patches = []
        for res in self._unclassifiedDetections:
            bounding_box = res['bounding_box']
            self.ax.add_patch(patches.Rectangle(
                    xy=(bounding_box[1] * self._width,
                        bounding_box[0] * self._height),
                    width=(bounding_box[3] - bounding_box[1]) * self._width,
                    height=(bounding_box[2] - bounding_box[0]) * self._height, 
                    linewidth=2,
                    edgecolor=colors[res['class_id']],
                    facecolor='none'
            ))


        self.ax.imshow(self.img)
        plt.draw()
        self.fig.canvas.flush_events()

    def get_detection_for_angle(self, angle) -> int:
        for detection in self._unclassifiedDetections:
            if abs(angle - detection['h_center']) <= CP_THRESHOLD:
                return detection['class_id']
        return -1

    def get_detection_for_centerpoint(self, centerpoint) -> int:
        offset = np.arctan(centerpoint[1] / centerpoint[0])
        for detection in self._unclassifiedDetections:
            if abs(offset - detection['h_center']) <= CP_THRESHOLD:
                return detection['class_id']
        return -1

    def run_detection(self, image):
        img = Image.fromarray(image).resize((self._width, self._height))
        input_data = np.expand_dims(img, axis=0)
        res = detect_objects(self._interpreter, input_data, self._threshold)
        for r in res:
            bbox = r['bounding_box']
            x_center = ((bbox[3] - bbox[1]) / 2 + bbox[1])
            # calculate how many degrees over
            r['h_center'] = (x_center - 0.5) * pi / 2
        self._unclassifiedDetections = res
        self.img = img
