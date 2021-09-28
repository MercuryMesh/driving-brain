from argparse import ArgumentParser
import pathlib
from airsim.client import CarClient
from VisionDelegate import VisionDelegate
from utils import get_image

if __name__ == '__main__':
    parser = ArgumentParser()
    cwd = pathlib.Path(__file__).parent.absolute()
    parser.add_argument(
        '-t',
        '--threshold',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '-m',
        '--model',
        default=f"{cwd}/../models/model3.tflite"
    )
    parser.add_argument(
        '-n',
        '--numthreads',
        type=int,
        default=32
    )
    args = parser.parse_args()

    client = CarClient()
    client.confirmConnection()
    client.enableApiControl(False)

    delegate = VisionDelegate(args.model, lambda: get_image(client), args.numthreads, args.threshold)
    delegate.start_debug()
