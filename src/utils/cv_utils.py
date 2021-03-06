from airsim.client import CarClient
import cv2
from airsim import ImageRequest, ImageType
import numpy as np

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
    'sign',
    'unknown'
]

def get_image(client: CarClient):
    image_response = client.simGetImages([ImageRequest(0, ImageType.Scene, False, False)])[0]
    arr = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
    image_bgr = arr.reshape(image_response.height, image_response.width, 3)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

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

def detect_objects(interpreter, image, threshold):
    set_input(interpreter, image)
    interpreter.invoke()

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
