from typing import Tuple
from airsim.client import CarClient
from airsim.types import LidarData
import cv2
from airsim import ImageRequest, ImageType
import numpy as np
from threading import Event, Thread, Lock
import numpy

colors = [
    'red',
    'green',
    'yellow',
    'black'
]

class_names = [
    'car',
    'pedestrian',
    'cyclist',
    'sign',
    'unknown'
]

airsimIOLock = Lock()

def calculateCenterPoint(group):
    npa = numpy.array(group)
    xyz = numpy.zeros(npa.shape[-1], dtype=npa.dtype)
    for point in group:
        xyz += point
    return xyz / len(group)

def calculateClosestPoint(group):
    m = map(lambda g: g[1], group)
    closest_offset = m.__next__()
    closest_index = 0
    for index, hoff in enumerate(m):
        if abs(hoff) < closest_offset:
            closest_offset = abs(hoff)
            closest_offset = index + 1
    return group[closest_index]

def parse_lidar_data(data: LidarData) -> numpy.ndarray:
    points = numpy.array(data.point_cloud, dtype=numpy.dtype('f4'))
    points = numpy.reshape(points, (int(points.shape[0] / 3), 3))
    # points = list(map(lambda p: {"point": p, "distance": distance_of_point(p)}, points))
    return points

def distance_of_point(to_point, from_point):
    sum = np.sum(np.square( from_point - to_point ))
    return np.sqrt(sum)
