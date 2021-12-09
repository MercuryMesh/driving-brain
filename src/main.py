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

# New (From bcwadsworth MQTT bit)
import paho.mqtt.client as mqtt

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

    try:
        # New (From bcwadsworth MQTT bit)
        client = mqtt.Client();
        client.on_connect = lambda client, userdata, flags, rc : print("Connected with result code "+ str(rc));
        client.on_message = lambda client, userdata, msg : print(msg.topic+" "+str(msg.payload));
        client.connect("localhost", 1883, 60);
        client.loop_start();
    except:
        pass


    args = parser.parse_args()
    carClient = CarClient()
    carClient.confirmConnection()
    carClient.enableApiControl(True)
    drivingArbiter = DrivingArbiter(carClient)
    visionDelegate = VisionDelegate(args.model, args.numthreads, args.threshold)
    laneDetection = LaneDetection(drivingArbiter)
    angularOccupancy = AngularOccupancy(client)
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
        angularOccupancy.sendobj();
        angularOccupancy.expire_occupants()
        angularOccupancy.draw()
