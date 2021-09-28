from airsim import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps
import numpy as np
import label_image
import cv2
import tflite_runtime.interpreter as tflite
import argparse

client = CarClient()
client.confirmConnection()
client.enableApiControl(False)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
plt.show()


def get_image():
    image_response = client.simGetImages([ImageRequest(0, ImageType.Scene, False, False)])[0]
    arr = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
    image_bgr = arr.reshape(image_response.height, image_response.width, 3)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        default='./model.tflite'
    )
    parser.add_argument(
        '-n',
        '--numthreads',
        type=int,
        default=32
    )
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=0.7
    )
    args = parser.parse_args()
    interpreter = tflite.Interpreter(model_path=args.model, num_threads=args.numthreads)
    interpreter.allocate_tensors()

    input_shape = interpreter.get_input_details()[0]['shape']
    height = input_shape[1]
    width = input_shape[2]

    while (True):
        plt.cla()
        img = Image.fromarray(get_image()).resize((width, height))
        input_data = np.expand_dims(img, axis=0)

        results = label_image.detect_objects(interpreter, input_data, args.threshold)
        print(list(map(lambda r: label_image.class_names[r["class_id"]], results)))

        ax.patches = []
        for res in results:
            bounding_box = res['bounding_box']
            ax.add_patch(patches.Rectangle(
                    xy=(bounding_box[1] * width,
                        bounding_box[0] * height),
                    width=(bounding_box[3] - bounding_box[1]) * width,
                    height=(bounding_box[2] - bounding_box[0]) * height, 
                    linewidth=2,
                    edgecolor=label_image.colors[res['class_id']],
                    facecolor='none'
            ))

        ax.imshow(img)
        plt.draw()
        fig.canvas.flush_events()
        