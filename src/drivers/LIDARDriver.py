from time import time
from airsim.types import LidarData
import numpy
from numpy.lib.arraysetops import isin
from drivers.Driver import DriverPriority, Driver
from drivers.DrivingArbiter import DrivingArbiter, SpeedController
from utils import distance_of_point, parse_lidar_data

HUMAN_STOP_DISTANCE = 3

DISTANCE_THRESHOLD = 5
def groupContiguousPoints(point_cloud):
    groups = []
    group = []
    in_group = False
    for i in range(0, len(point_cloud), 2):
        if not in_group:
            group = [point_cloud[i]]
            in_group = True
        elif distance_of_point(point_cloud[i], group[len(group) - 1]) <= DISTANCE_THRESHOLD:
            group.append(point_cloud[i])
        else:
            groups.append(group)
            in_group = False
    return groups

def calculateCenterPoint(group):
    xyz = [0.0, 0.0, 0.0]
    for point in group:
        xyz += point
    return xyz / len(group)
    
SIDE_REGION = 3
def isInFront(point):
    return abs(point[1]) <= SIDE_REGION and point[0] >= 0

class LIDARDriver(Driver):
    id = "lidar-driver"
    _speedController: SpeedController = None

    def __init__(self, drivingArbiter: DrivingArbiter):
        self._drivingArbiter = drivingArbiter
        drivingArbiter.requestSpeedControl(self, DriverPriority.Low, returnControl=False)

    def onSpeedGranted(self, controller):
        self._speedController = controller
        return super().onSpeedGranted(controller)

    def onSpeedRevoked(self, willReturn):
        self._speedController = None
        return super().onSpeedRevoked(willReturn)

    def onSteeringGranted(self, controller):
        return super().onSteeringGranted(controller)

    def onSteeringRevoked(self, willReturn):
        return super().onSteeringRevoked(willReturn)
    
    def setSpeed(self, lidarData: LidarData, currentSpeed: float):
        points = parse_lidar_data(lidarData)
        grouped = groupContiguousPoints(points)
        
        attentionDistances = []
        for group in grouped:
            center = calculateCenterPoint(group)
            if isInFront(center):
                attentionDistances.append(center[0])
        
        if len(attentionDistances) == 0:
            if currentSpeed <= 15:
                self._speedController.brake = 0
                self._speedController.throttle = 2
            else:
                self._speedController.brake = 0
                self._speedController.throttle = 0
            return

        closest = min(attentionDistances) - ((currentSpeed / 2) ** 2)
        self._speedController.throttle

        self._speedController.throttle = min(max(closest - 10, 0), 1 if currentSpeed <= 15 else 0)
        self._speedController.brake = max(0, 15 - closest)
                