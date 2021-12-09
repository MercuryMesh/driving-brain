from airsim.client import CarClient
from airsim.types import LidarData
import numpy as np

def parse_lidar_data(data: LidarData) -> np.ndarray:
    points = np.array(data.point_cloud, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 3), 3))
    # points = list(map(lambda p: {"point": p, "distance": distance_of_point(p)}, points))
    return points
carClient = CarClient()
carClient.confirmConnection()
carClient.enableApiControl(True)

lidarSpeedData = carClient.getLidarData(lidar_name='MyLidar2')
print(parse_lidar_data(lidarSpeedData))