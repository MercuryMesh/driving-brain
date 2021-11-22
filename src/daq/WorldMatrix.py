from typing import Dict, List, Tuple
import numpy as np
from uuid import uuid4
from utils import distance_of_point

LIDAR_RANGE = 168
DISTANCE_THRESHOLD = 3
normalizer = int((LIDAR_RANGE * 10))

class WorldObject:
    classification: int
    direction_vector: Tuple[int, int]
    speed: int
    center_coord: Tuple[int, int]

    def __init__(self, center_coord, classification = -1, direction_vector = None, speed = None):
        self.center_coord = center_coord
        self.classification = classification
        self.direction_vector = direction_vector
        self.speed = speed

# The world matrix holds a 2D representation of the known area
# Each cell represents 1/10 of a meter
# The vehicle is always in the center of matrix
# X is the horizontal axis, Y is the vertical (or straight forward axis)
class WorldMatrix:
    object_map: Dict[int, WorldObject]
    speed: int
    direction_vector: Tuple[int, int]

    def __init__(self):
        self.matrix = np.zeros((LIDAR_RANGE * 2 * 10, LIDAR_RANGE * 2 * 10))
        self.object_map = {}
        self.speed = 0
        self.direction_vector = (0, 0)

    def set_direction(self, steering_angle):
        x = np.sin(steering_angle)
        y = np.cos(steering_angle)
        self.direction_vector = (x, y)
    
    def set_speed(self, speed):
        self.speed = speed

    def set_from_detection_frame(self, detections: List):
        matrix = np.zeros((LIDAR_RANGE * 2 * 10, LIDAR_RANGE * 2 * 10))
        for d in detections:
            self.add_detected_object(d, matrix)
        self.matrix = matrix

    def check_if_known(self, obj):
        check_x = obj.centerPoint[1] * 10 + normalizer
        check_y = obj.centerPoint[0] * 10 + normalizer

        min_x = int(check_x - DISTANCE_THRESHOLD * 10)
        max_x = int(check_x + DISTANCE_THRESHOLD * 10)
        min_y = int(check_y - DISTANCE_THRESHOLD * 10)
        max_y = int(check_y + DISTANCE_THRESHOLD * 10)

        if (min_x < 0 or min_y < 0) or (max_x > LIDAR_RANGE * 2 * 10 or max_y > LIDAR_RANGE * 2 * 10):
            return None

        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                if self.matrix[i][j] != 0:
                    return self.matrix[i][j]
        return None

    def add_detected_object(self, obj, to_matrix):
        known_id = self.check_if_known(obj)
        cp_reframed = (obj.centerPoint[1] * 10, obj.centerPoint[0] * 10)
        if known_id is not None and known_id in self.object_map:
            known_obj = self.object_map[known_id]

            old_cp = known_obj.center_coord
            dir_vec = cp_reframed - old_cp
            speed = distance_of_point(cp_reframed, old_cp)
            normalized_dir = (dir_vec) / speed
            known_obj.center_coord = cp_reframed
            known_obj.direction_vector = normalized_dir
            known_obj.speed = speed
            self.object_map[known_id] = known_obj
           
        np_blob = np.array(obj.blob)
        x = np_blob[:, 1:]
        y = np_blob[:, :1]
        minx = int(np.min(x) * 10) + normalizer
        maxx = int(np.max(x) * 10) + normalizer
        miny = int(np.min(y) * 10) + normalizer
        maxy = int(np.max(y) * 10) + normalizer
        if (minx < 0 or miny < 0) or (maxx > LIDAR_RANGE * 2 * 10 or maxy > LIDAR_RANGE * 2 * 10):
            return

        obj_id = known_id if known_id is not None else uuid4().int
        for i in range(minx, maxx + 1):
            for j in range(miny, maxy + 1):
                to_matrix[i][j] = obj_id