import pathlib
from argparse import ArgumentParser
from time import time

from airsim.client import CarClient

from daq.LidarDelegate import LidarDelegate
from daq.VisionDelegate import VisionDelegate
from drivers.CollisionWatchdog import CollisionWatchdog
from drivers.DrivingArbiter import DrivingArbiter
from drivers.LaneDetection import LaneDetection
from managers.AngularOccupancy import AngularOccupancy
from utils.cv_utils import get_image
from utils.RerunableThread import RerunableThread

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
    angularOccupancy = AngularOccupancy()
    lidarDriver = LidarDelegate(angularOccupancy)
    collisionWatchdog = CollisionWatchdog(drivingArbiter, angularOccupancy)

    laneDetection.start()
    maxLoopDelay = 0.0
    lastLoop = time()
    # laneThread = RerunableThread(laneDetection.follow_lane)
    # lidarThread = RerunableThread(lidarDriver.checkLidar, name="lidarThread1")
    # collisionThread = RerunableThread(collisionWatchdog.runLoop, name="collisionThread1")
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
        lidarDriver.checkLidar(lidarData)
        collisionWatchdog.runLoop(currentSpeed)
        laneDetection.follow_lane(img, currentSpeed)
        angularOccupancy.expire_occupants()
        angularOccupancy.draw()
