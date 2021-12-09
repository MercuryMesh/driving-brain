import numpy as np
from typing import Tuple
from time import time_ns
from utils.lidar_utils import distance_of_point
from threading import Timer

DECAY_FACTOR = 0.75
DECAY_BIAS = 0.05
DECAY_DELAY = 0.5

DANGER_REGION = 1

ROC_UPDATE_TIME = 1
class Occupant:
    center_point: Tuple[np.float32, np.float32]
    center_angle: np.float32
    distance: np.float32
    relative_speed: np.float32
    relative_velocity: Tuple[np.float32, np.float32]

    classification: int
    
    weight: np.float32
    weight_roc: np.float32
    probability: np.float32

    _last_update_time: int

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
        self.probability = 0.75
        self._last_update_time = time_ns()
        self.relative_velocity = (0, 0)
        self.weight = 0.0
        
        self._timer = Timer(DECAY_DELAY, self._decay)
        self._timer.start()

    def _decay(self):
        self.probability = (self.probability * DECAY_FACTOR) - DECAY_BIAS
        self._timer = Timer(DECAY_DELAY, self._decay)
        self._timer.start()

    def kill(self):
        self._timer.cancel()

    def update_with(self, other):
        self.update(other.classification, other.center_point)

    def update(self, classification = None, center_point = None):
        update_time = time_ns()
        time_delta = update_time - self._last_update_time
        if time_delta < ROC_UPDATE_TIME:
            return
        self.probability = min(1, self.probability * (1 + DECAY_FACTOR) + DECAY_BIAS)

        if classification is not None:
            self.classification = classification
        if center_point is not None:
            new_distance = distance_of_point(center_point, (0, 0))
            self.relative_speed = (self.distance - new_distance) * 1e9 / (time_delta)
            self.relative_velocity = (center_point - self.center_point) * 1e9 / (time_delta)
            self.distance = new_distance
            self.center_point = center_point

        new_weight = self.weigh()
        self.weight_roc = (new_weight - self.weight) / (time_delta)
        self._last_update_time = time_ns()
        self.weight = new_weight

    def set_weight(self, new_weight):
        self.weight = new_weight
    
    def weigh(self) -> float:
        return self.weight
        w = 0
        if abs(self.center_angle) < (np.pi / 8):
            bias = 5
            s_check = 3
            w += 10 / self.distance
        else:
            bias = 2
            s_check = 2

        for i in range(1, 101):
            s_off = s_check/i
            predicted_location = (self.center_point[0] + self.relative_velocity[0] * s_off, self.center_point[1] + self.relative_velocity[1] * s_off)
            if abs(predicted_location[0]) < DANGER_REGION and predicted_location[1] < DANGER_REGION:
                w += (bias - s_off)

        return (w ) * self.probability