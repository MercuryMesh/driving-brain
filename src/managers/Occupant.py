import numpy as np
from typing import Tuple
from time import time
from utils.lidar_utils import distance_of_point
from threading import Timer

DECAY_FACTOR = 0.8
DECAY_BIAS = 0.05
DECAY_DELAY = 0.4

class Occupant:
    center_point: Tuple[np.float32, np.float32]
    center_angle: np.float32
    distance: np.float32
    relative_speed: np.float32

    classification: int
    
    weight: np.float32
    weight_roc: np.float32
    probability: np.float32

    _last_update_time: float
    _last_speed_update_time: float

    def __init__(self, center_point, center_angle = None, classification = -1):
        self.classification = classification
        self.relative_speed = 0
        self.center_point = center_point
        if center_angle is not None:
            self.center_angle = center_angle
        else:
            self.center_angle = np.arctan2(center_point[0], center_point[1])

        self.distance = distance_of_point(center_point, (0, 0))
        self.weight_roc = 0
        self.probability = 1
        self._last_update_time = time()
        self._last_speed_update_time = time()
        self.weight = self.weigh()
        
        self._timer = Timer(DECAY_DELAY, self._decay)
        self._timer.start()

    def _decay(self):
        self.probability = (self.probability * DECAY_FACTOR) - DECAY_BIAS
        self._timer = Timer(DECAY_DELAY, self._decay)
        self._timer.start()

    def kill(self):
        self._timer.cancel()

    def update(self, classification = None, center_point = None):
        update_time = time()
        time_delta = update_time - self._last_update_time
        if time_delta == 0:
            return
        self._last_update_time = update_time
        self.probability = 1

        if classification is not None:
            self.classification = classification
        if center_point is not None:
            new_distance = distance_of_point(center_point, (0, 0))
            t = time()
            if t - self._last_speed_update_time >= 0.5:
                self.relative_speed = (self.distance - new_distance) / (t - self._last_speed_update_time)
                self._last_speed_update_time = t
            self.distance = new_distance
            self.center_point = center_point

        new_weight = self.weigh()
        self.weight_roc = (new_weight - self.weight) / (time_delta)
        self.weight = new_weight
    
    def weigh(self) -> float:
        return ((10 / self.distance) + self.relative_speed)
