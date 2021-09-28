import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model_file',
        help='.tflite model to be executed'
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        help='number of threads',
        default=None
    )
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        help='image to be classified'
    )
    parser.add_argument(
      '--input_mean',
      default=127.5, type=float,
      help='input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type=float,
        help='input standard deviation')
    args = parser.parse_args()

    interpreter = tflite.Interpreter(model_path=args.model_file, num_threads=args.num_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    floating_model = input_details[0]['dtype'] == np.float32

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    img = Image.open(args.input).resize((width, height))
    
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - args.input_mean) / args.input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    for i in top_k:
        if floating_model:
            print('{:08.6f}: {}'.format(float(results[i])))
        else:
            print('{:08.6f}: {}'.format(float(results[i] / 255.0)))
