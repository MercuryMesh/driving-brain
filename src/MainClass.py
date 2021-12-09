from time import time
from airsim.client import CarClient
import numpy

from daq.LidarDelegate import LidarDelegate
from drivers.CollisionWatchdog import CollisionWatchdog
from drivers.DrivingArbiter import DrivingArbiter
from drivers.LaneDetection import LaneDetection
from managers.AngularOccupancy import AngularOccupancy
from utils.cv_utils import get_image
from utils.RerunableThread import RerunableThread

class Main:
    def __init__(self, carClient: CarClient, collisionWatchdog: CollisionWatchdog, drivingArbiter: DrivingArbiter, angularOccupancy: AngularOccupancy):
        self.carClient = carClient
        self.carClient.confirmConnection()
        self.carClient.enableApiControl(True)
        self.drivingArbiter = drivingArbiter
        self.laneDetection = LaneDetection(self.drivingArbiter)
        self.laneDetection.start()

        self.angularOccupancy = angularOccupancy
        self.lidarDriver = LidarDelegate(self.angularOccupancy)
        self.collisionWatchdog = collisionWatchdog
        
    def main(self):
        self.drivingArbiter.sendBatch()
        img = get_image(self.carClient)
        lidarData = self.carClient.getLidarData('MyLidar1')
        currentSpeed = self.carClient.getCarState().speed
        self.lidarDriver.checkLidar(lidarData)
        self.collisionWatchdog.runLoop(currentSpeed)
        self.laneDetection.follow_lane(img, currentSpeed)
        self.angularOccupancy.expire_occupants()

