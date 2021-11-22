import enum
import sys
from math import pi
from time import sleep, time
from typing import List, Tuple
from airsim.types import GeoPoint, LidarData
import numpy
import matplotlib.pyplot as plt
from daq.WorldMatrix import WorldMatrix
from drivers.Driver import DriverPriority, Driver
from drivers.DrivingArbiter import DrivingArbiter, SpeedController
from utils import distance_of_point, parse_lidar_data, calculateCenterPoint, class_names

HUMAN_STOP_DISTANCE = 10

DISTANCE_THRESHOLD = 3
MIN_POINTS_IN_GROUP = 2
def groupContiguousPoints(point_cloud):
    groups = []
    group = [point_cloud[0]]
    for point in point_cloud[1:]:
        dis = distance_of_point(point, group[-1])
        # print(dis)
        if dis <= DISTANCE_THRESHOLD:
            group.append(point)
        else:
            if len(group) >= MIN_POINTS_IN_GROUP:
                groups.append(group)
            group = [point]
    return groups

def group_to_detected_object(group):
    cp = calculateCenterPoint(group)
    return DetectedObject(group, cp)
    
SIDE_REGION = 3
def isInFront(point):
    return abs(point[1]) <= SIDE_REGION and point[0] >= 0

def isInCameraView(point):
    if point[0] <= 0:
        return False
    
    # detection doesn't work when too far away
    dis = distance_of_point((0, 0), point)
    if dis > 30:
        return False

    angle = numpy.arctan(point[1] / point[0])
    return abs(angle) <= pi / 4

class DetectedObject(object):
    blob: List[List[float]]
    centerPoint: List[float]
    classification: int

    def __init__(self, blob, centerPoint, classification = -1):
        self.blob = blob
        self.centerPoint = centerPoint
        self.classification = classification

class LIDARDriver(Driver):
    id = "lidar-driver"
    _lastSlow = -1
    _speedController: SpeedController = None

    def __init__(self, drivingArbiter: DrivingArbiter, worldMatrix: WorldMatrix):
        self._drivingArbiter = drivingArbiter
        self._worldMatrix = worldMatrix
        # drivingArbiter.requestSpeedControl(self, DriverPriority.Low, returnControl=False)

    @staticmethod
    def flatten(data: numpy.array):
        flattened = data[:, :2]
        return numpy.unique(flattened, axis=0)

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

    def debugLidar(self, lidarData: LidarData, img):
        points = parse_lidar_data(lidarData)
        points = LIDARDriver.flatten(points)
        grouped = groupContiguousPoints(points)

        plt.figure()
        plt.imshow(img)
        forward_groups = list(filter(lambda x: isInCameraView(calculateCenterPoint(x)), grouped))

        colors = [
            "red",
            "blue",
            "green",
            "orange",
            "yellow",
            "pink",
            "blue",
            "red",
            "green"
        ]
        self._visionDelegate.run_detection(img)
        for i,g in enumerate(forward_groups):
            leftest = min(g, key = lambda k: k[1])
            rightest = max(g, key = lambda k: k[1])
            l_angle = numpy.arctan(leftest[1] / leftest[0]) + (numpy.pi / 4)
            r_angle = numpy.arctan(rightest[1] / rightest[0]) + (numpy.pi / 4)

            l_pct = l_angle / (numpy.pi/2)
            r_pct = r_angle / (numpy.pi/2)

            l_pt = (l_pct * len(img), r_pct * len(img))
            r_pt = (len(img[0]) / 2, len(img[0]) / 2)
            plt.plot(l_pt, r_pt, color=colors[i], linewidth=4)
            t = class_names[self._visionDelegate.get_detection_for_centerpoint(calculateCenterPoint(g))]
            plt.text(l_pt[0], r_pt[0], t)
        plt.show()
    
    def checkLidar(self, lidarData: LidarData, currentSpeed: float):
        points = parse_lidar_data(lidarData)
        points = LIDARDriver.flatten(points)
        if len(points) == 0:
            return
        grouped = groupContiguousPoints(points)
        objects = list(map(group_to_detected_object, grouped))
        self._worldMatrix.set_from_detection_frame(objects)

        attentionDistances = []
        for obj in objects:
            center = obj.centerPoint
            if isInFront(center) and center[0] <= 30:
                attentionDistances.append(center[0])

        if len(attentionDistances) > 0:
            m = min(attentionDistances)
            closest = m - ((currentSpeed) ** 2)
            if closest <= HUMAN_STOP_DISTANCE:
                priority = DriverPriority.Crucial
            elif closest <= HUMAN_STOP_DISTANCE * 1.5:
                priority = DriverPriority.High
            else:
                priority = DriverPriority.Medium

            if self._drivingArbiter.requestSpeedControl(self, priority):
                self._speedController.throttle
                self._speedController.throttle = min(max(closest - HUMAN_STOP_DISTANCE - (currentSpeed / 2), 0), 0)
                self._speedController.brake = max(0, HUMAN_STOP_DISTANCE + (HUMAN_STOP_DISTANCE - (closest / 2)) + (currentSpeed / 2))
        else:
            self._drivingArbiter.giveUpSpeedControl(self)
                