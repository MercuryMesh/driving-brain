import tflite_runtime.interpreter as tflite
from utils import ThreadWithReturnValue, colors, set_input, detect_objects
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class VisionDelegate:
    debug_initialized = False
    canTakeFrame = True
    thread: ThreadWithReturnValue = None
    img = None
    
    def __init__(self, model_path, get_image, numthreads=32, threshold=0.5):
        self._threshold = threshold
        self._interpreter = tflite.Interpreter(model_path=model_path, num_threads=numthreads)
        self._interpreter.allocate_tensors()
        self._get_image = get_image

        input_shape = self._interpreter.get_input_details()[0]['shape']
        self._height = input_shape[1]
        self._width = input_shape[2]
    
    def start_debug(self):
        self.debug_running = True
        while(self.debug_running):
            self.debug_label()

    def show_processed_img(self):
        results = self.thread.join()
        self._ax.patches = []
        for res in results:
            bounding_box = res['bounding_box']
            self._ax.add_patch(patches.Rectangle(
                xy=(bounding_box[1] * self._width,
                    bounding_box[0] * self._height),
                width=(bounding_box[3] - bounding_box[1]) * self._width,
                height=(bounding_box[2] - bounding_box[0]) * self._height,
                linewidth=2,
                edgecolor=colors[res['class_id']],
                facecolor='none'
            ))
        
        self._ax.imshow(self.img)
        plt.draw()
        self._fig.canvas.flush_events()
        self.canTakeFrame = True


    def debug_thread(self, image):
        if self.canTakeFrame:
            self.debug_label(image)
        elif not self.thread.is_alive():
            self.show_processed_img()

    def debug_label(self, image):
        self.canTakeFrame = False
        if not self.debug_initialized:
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111)
            plt.ion()
            plt.show()
            self.debug_initialized = True
        
        plt.cla()

        img = Image.fromarray(image).resize((self._width, self._height))
        input_data = np.expand_dims(img, axis=0)
        self.img = img
        
        # spawn a thread to process the image
        detect_thread = ThreadWithReturnValue(target=detect_objects, args=(self._interpreter, input_data, self._threshold ))
        detect_thread.start()
        self.thread = detect_thread
