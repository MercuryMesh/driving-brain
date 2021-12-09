
from math import pi
from time import sleep, time
from airsim.types import GeoPoint, LidarData
import numpy
from managers.AngularOccupancy import AngularOccupancy
from utils.lidar_utils import distance_of_point, parse_lidar_data, groupContiguousPoints

HUMAN_STOP_DISTANCE = 15
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

class LidarDelegate:
    _lastSlow = -1

    def __init__(self, angularOccupancy: AngularOccupancy):
        self._angularOccupancy = angularOccupancy
        # drivingArbiter.requestSpeedControl(self, DriverPriority.Low, returnControl=False)

    @staticmethod
    def flatten(data: numpy.array):
        flattened = data[:, :2]
        return numpy.unique(flattened, axis=0)

    def checkLidar(self, lidarData: LidarData):
        points = parse_lidar_data(lidarData)
        points = LidarDelegate.flatten(points)
        if len(points) == 0:
            return
        grouped = groupContiguousPoints(points)
        self._angularOccupancy.occupant_from_blobs(grouped)

                