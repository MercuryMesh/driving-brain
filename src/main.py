import pathlib
from argparse import ArgumentParser
from time import time

from airsim.client import CarClient
import numpy

from daq.LidarDelegate import LidarDelegate
from daq.VisionDelegate import VisionDelegate
from drivers.CollisionWatchdog import CollisionWatchdog
from drivers.DrivingArbiter import DrivingArbiter
from drivers.LaneDetection import LaneDetection
from managers.AngularOccupancy import AngularOccupancy
from utils.cv_utils import get_image
from utils.RerunableThread import RerunableThread
from pstats import SortKey

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

    # New (From bcwadsworth MQTT bit)
    client = mqtt.Client();
    client.on_connect = lambda client, userdata, flags, rc : print("Connected with result code "+ str(rc));
    client.on_message = lambda client, userdata, msg : print(msg.topic+" "+str(msg.payload));
    client.connect("localhost", 1883, 60);
    client.loop_start();


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
    lastLoop = time()
    laneThread = RerunableThread(laneDetection.follow_lane)

    delays = numpy.zeros(50)
    i = 0
    while True:
        if i == 50:
            a = numpy.average(delays)
            print(f"Average loop time {a}s")
            i = 0
        loopDelay = time() - lastLoop
        delays[i] = loopDelay
        i += 1
        lastLoop = time()
        drivingArbiter.sendBatch()
        img = get_image(carClient)
        lidarData = carClient.getLidarData('MyLidar1')
        currentSpeed = carClient.getCarState().speed
        lidarDriver.checkLidar(lidarData)
        collisionWatchdog.runLoop(currentSpeed)
        if not laneThread.is_running:
            laneThread.run((img, currentSpeed))
        angularOccupancy.expire_occupants()
        # angularOccupancy.draw()
        angularOccupancy.sendobj()
