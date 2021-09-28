from PIL import Image
from PIL import ImageFilter
from PIL.ImageFilter import Filter
import tflite_runtime.interpreter as tflite
import argparse
import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

colors = [
    'red',
    'green',
    'yellow',
    'black'
]

class_names = [
    'car',
    'pedestrian',
    'cyclist',
    'sign'
]

def set_input(interpreter, input_data, input_mean=127.5, input_std=127.5):
    input_details = interpreter.get_input_details()[0]
    if input_details['dtype'] == np.float32:
        input_data = (np.float32(input_data) - input_mean) / input_std

    tensor_index = input_details['index']
    interpreter.set_tensor(tensor_index, input_data)

def get_output(interpreter, index):
    tensor_index = interpreter.get_output_details()[index]['index']
    output_data = interpreter.get_tensor(tensor_index)
    return np.squeeze(output_data)

def detect_objects(interpreter, image, threshold, time_exec=False):
    set_input(interpreter, image)
    if time_exec:
        start_time = time.time()
    interpreter.invoke()
    if time_exec:
        end_time = time.time()
        print(f"Model ran in {end_time - start_time}s")

    boxes = get_output(interpreter, 0)
    classes = get_output(interpreter, 1)
    scores = get_output(interpreter, 2)
    count = int(get_output(interpreter, 3))
    results = []
    for i in range(count):
        # print(f"Detected a {class_names[int(classes[i])]} with confidence {scores[i]}")
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': int(classes[i]),
                'score': scores[i]
            }
            results.append(result)


    return results

if __name__ ==  '__main__':
    script_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        help="image to be classified"
    )
    parser.add_argument(
        '-t',
        '--threshold',
        default=0.4
    )
    parser.add_argument(
        '-m',
        '--model',
        default='./model.tflite'
    )

    args = parser.parse_args()
    interpreter = tflite.Interpreter(model_path=args.model, num_threads=64)
    interpreter.allocate_tensors()

    input_shape = interpreter.get_input_details()[0]['shape']

    height = input_shape[1]
    width = input_shape[2]

    img = Image.open(args.image).convert('RGB').resize((width, height))
    input_data = np.expand_dims(img, axis=0)

    results = detect_objects(interpreter, input_data, float(args.threshold), time_exec=True)

    fig, ax = plt.subplots()
    for res in results:
        bounding_box = res['bounding_box']
        ax.add_patch(patches.Rectangle(
                xy=(bounding_box[1] * width,
                    bounding_box[0] * height),
                width=(bounding_box[3] - bounding_box[1]) * width,
                height=(bounding_box[2] - bounding_box[0]) * height, 
                linewidth=2,
                edgecolor=colors[res['class_id']],
                facecolor='none'
            ))

    print(f"Whole script executed in {time.time() - script_start} s")
    plt.imshow(img)
    plt.show()