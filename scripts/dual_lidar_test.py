from airsim.client import CarClient
from airsim.types import LidarData
import numpy as np

def parse_lidar_data(data: LidarData, extra: LidarData = None) -> np.ndarray:
    if extra is not None:
        points = np.array(list(zip(data.point_cloud, extra.point_cloud)), dtype=np.float32)
        points = np.reshape(points, (points.shape[0] // 3, 3, 2))
    else:
        points = np.array(data.point_cloud, dtype=np.float32)
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
    # points = list(map(lambda p: {"point": p, "distance": distance_of_point(p)}, points))
    return points

def flatten(data: np.array):
        flattened = data[:, :2, :]
        return np.unique(flattened, axis=0)
carClient = CarClient()
carClient.confirmConnection()
carClient.enableApiControl(True)

lidarDistanceData = carClient.getLidarData(lidar_name='MyLidar1')
lidarSpeedData = carClient.getLidarData(lidar_name='MyLidar2')
print(np.array(lidarDistanceData.point_cloud))

print("=================\n\n")
print(np.array(lidarSpeedData.point_cloud))
print("=================\n\n")

print(np.array(list(zip(lidarDistanceData.point_cloud, lidarSpeedData.point_cloud))))

print("=================\n\n")
d = parse_lidar_data(lidarDistanceData, lidarSpeedData)
print(d)
print("================\n\n")
d = flatten(d)
print(d)
