from argparse import ArgumentParser
import pathlib
from threading import Thread
from airsim import client
from airsim.client import CarClient
from msgpackrpc import loop
from VisionDelegate import VisionDelegate
from drivers.DrivingArbiter import DrivingArbiter
from drivers.LIDARDriver import SIDE_REGION, LIDARDriver
from drivers.LaneDetection import LaneDetection
from utils import RerunableThread, calculateCenterPoint, get_image, class_names, parse_lidar_data
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
    visionDelegate = VisionDelegate(args.model, args.numthreads, args.threshold)
    laneDetection = LaneDetection(drivingArbiter)
    lidarDriver = LIDARDriver(drivingArbiter, visionDelegate)
    laneDetection.start()

    maxLoopDelay = 0.0
    lastLoop = time()
    laneThread = RerunableThread(laneDetection.follow_lane)
    lidarThread = RerunableThread(lidarDriver.checkLidar, name="lidarThread1")
    while True:
        loopDelay = time() - lastLoop
        if loopDelay > maxLoopDelay:
            print(f"Max loop delay {loopDelay}s")
            maxLoopDelay = loopDelay
        lastLoop = time()
        drivingArbiter.sendBatch()
        img = get_image(carClient)
        lidarData = carClient.getLidarData()
        currentSpeed = carClient.getCarState().speed
        # lidarDriver.debugLidar(lidarData)
        # lidarDriver.checkLidar(lidarData, currentSpeed, img)
        # if not laneThread.is_running:
        #     laneThread.run((img, currentSpeed,))
        if not lidarThread.is_running:
            lidarThread.run((lidarData, currentSpeed, img))
