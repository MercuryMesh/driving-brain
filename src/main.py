from argparse import ArgumentParser
import pathlib
from airsim.client import CarClient
from VisionDelegate import VisionDelegate
from drivers.DrivingArbiter import DrivingArbiter
from drivers.LaneDetection import LaneDetection
from utils import get_image
from time import time

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
    carClient = CarClient()
    carClient.confirmConnection()
    carClient.enableApiControl(True)
    drivingArbiter = DrivingArbiter(carClient)
    visionDelegate = VisionDelegate(args.model, lambda: get_image(carClient), args.numthreads, args.threshold)
    laneDetection = LaneDetection(drivingArbiter, carClient)
    laneDetection.start()
    
    ### Image loop
    last_loop = time()
    while True:
        print(f"loop delay: {time() - last_loop}")
        last_loop = time()
        img = get_image(carClient)
        laneDetection.follow_lane(img)
        visionDelegate.debug_thread(img)