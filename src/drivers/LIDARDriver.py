import enum
import sys
from math import pi
from time import time
from typing import List, Tuple
from airsim.types import LidarData
import numpy
import matplotlib.pyplot as plt
from numpy.lib.arraysetops import isin
from VisionDelegate import VisionDelegate
from drivers.Driver import DriverPriority, Driver
from drivers.DrivingArbiter import DrivingArbiter, SpeedController
from utils import calculateClosestPoint, distance_of_point, parse_lidar_data, calculateCenterPoint, class_names

HUMAN_STOP_DISTANCE = 3

DISTANCE_THRESHOLD = 10.0
def groupContiguousPoints(point_cloud):
    groups = []
    group = [point_cloud[0]]
    in_group = False
    for point in point_cloud[1:]:
        dis = distance_of_point(point, group[-1])
        # print(dis)
        if dis <= DISTANCE_THRESHOLD:
            group.append(point)
        else:
            groups.append(group)
            group = [point]
    return groups
    
SIDE_REGION = 3
def isInFront(point):
    return abs(point[1]) <= SIDE_REGION and point[0] >= 0

def isInCameraView(point):
    if point[0] <= 0:
        return False
    
    # detection doesn't work when too far away
    dis = distance_of_point((0, 0), point)
    if dis > DISTANCE_THRESHOLD * 4:
        return False

    angle = numpy.arctan(point[1] / point[0])
    return abs(angle) <= pi / 4

class DetectedObject(object):
    blob: List[List[float]]
    centerPoint: List[float]
    classification: int
    stale: bool

    def __init__(self, blob, centerPoint, classification = -1):
        self.blob = blob
        self.centerPoint = centerPoint
        self.classification = classification
        self.stale = False

class LIDARDriver(Driver):
    id = "lidar-driver"
    _lastSlow = -1
    _speedController: SpeedController = None
    _knownObjects: List[DetectedObject] = None
    _lastCheckTime = 0

    def __init__(self, drivingArbiter: DrivingArbiter, visionDelegate: VisionDelegate):
        self._drivingArbiter = drivingArbiter
        self._visionDelegate = visionDelegate
        self._knownObjects = []
        self._lastCheckTime = time()
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

    def debugLidar(self, lidarData: LidarData):
        points = parse_lidar_data(lidarData)
        points = LIDARDriver.flatten(points)
        closest_distance = sys.float_info.max
        furthest_distance = 0.0

        # identify closest and furthest points
        # closest -> pure white #ffffff
        # furthest -> black #000000
        for point in points:
            dis = distance_of_point(point, (0, 0))
            if dis < closest_distance:
                closest_distance = dis
            if dis > furthest_distance:
                furthest_distance = dis
        
        distance_range = furthest_distance - closest_distance
        plt.figure()
        for point in points:
            if point[0] < 0:
                continue
            point_dis = distance_of_point(point, (0, 0))
            distance_pct = (point_dis - closest_distance) / distance_range
            color_val = 1 - distance_pct * 0.5
            plt.scatter(point[1], 0.0, color=(color_val, color_val, color_val))
        grouped = groupContiguousPoints(points)
        print(f"Identified {len(grouped)} groups")
        i = 1
        for group in grouped:
            center = calculateCenterPoint(group)
            if center[0] < 0:
                continue
            plt.annotate(f"({center[0]} {center[1]})", (center[1], -0.01 * i))
            i += 1
        print(f"{i} total loops")
        plt.show()
        # grouped = groupContiguousPoints(points)
    
    def checkLidar(self, lidarData: LidarData, currentSpeed: float, img):
        points = parse_lidar_data(lidarData)
        points = LIDARDriver.flatten(points)
        grouped = groupContiguousPoints(points)
        
        distance_moved = currentSpeed * (time() - self._lastCheckTime)
        self._lastCheckTime = time()
        # make our known data stale
        new = 0
        known = 0
        for group in grouped:
            cp = calculateCenterPoint(group)
            knownObjI = -1
            for i, obj in enumerate(self._knownObjects):
                if distance_of_point(cp, obj.centerPoint) <= distance_moved + DISTANCE_THRESHOLD * 5:
                    # print(f"We already know about that {class_names[self._knownObjects[i].classification]}")
                    knownObjI = i
                    break
            if knownObjI == -1:
                new += 1
                self._knownObjects.append(
                    DetectedObject(group, cp)
                )
            else:
                known += 1
                # if it's updated, it's not stale anymore
                self._knownObjects[i].blob = group
                self._knownObjects[i].centerPoint = cp
                self._knownObjects[i].stale = False

        self._knownObjects = list(filter(lambda x: not x.stale, self._knownObjects))
        attentionDistances = []
        has_classifiable = any(isInCameraView(obj.centerPoint) for obj in self._knownObjects)
        if has_classifiable:
            self._visionDelegate.run_detection(img)

        for i, obj in enumerate(self._knownObjects):
            center = obj.centerPoint
            if isInFront(center) and center[0] <= 20:
                attentionDistances.append(center[0])
            
            # if we don't know what it is, and it's in view and we have 
            if obj.classification == -1 and isInCameraView(center):
                det = self._visionDelegate.get_detection_for_centerpoint(center)
                self._knownObjects[i].classification = det


        if len(attentionDistances) > 0:
            m = min(attentionDistances)
            i = attentionDistances.index(m)
            closest = m - ((currentSpeed / 2) ** 2)
            if closest <= 10:
                priority = DriverPriority.Crucial
            elif closest <= 15:
                priority = DriverPriority.High
            else:
                priority = DriverPriority.Medium

            if self._drivingArbiter.requestSpeedControl(self, priority):
                self._speedController.throttle
                self._speedController.throttle = min(max(closest - 10, 0), 0)
                self._speedController.brake = max(0, 15 - closest)
        else:
            self._drivingArbiter.giveUpSpeedControl(self)
                