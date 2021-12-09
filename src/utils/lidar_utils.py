from airsim.types import LidarData
import numpy as np

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


def calculateCenterPoint(group):
    npa = np.array(group)
    xyz = np.zeros(npa.shape[-1], dtype=npa.dtype)
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

def parse_lidar_data(data: LidarData) -> np.ndarray:
    points = np.array(data.point_cloud, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 3), 3))
    # points = list(map(lambda p: {"point": p, "distance": distance_of_point(p)}, points))
    return points

def distance_of_point(to_point, from_point ):
    sum = np.sum(np.square( from_point - to_point ))
    return np.sqrt(sum)
