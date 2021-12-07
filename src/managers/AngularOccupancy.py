# This class creates an angular occupancy list relative to the current vehicle
# Occupied angles point to an object with details on the occupant
# Occupant details:
#      Distance from vehicle
#      Classification
#      Predicted relative velocity
#      Danger weight
#      Weight rate of change
# Weight implies how strictly an occupant should be avoided. It will be calculated 
# with the occupant details.
# New blobs are first passed through a movement filter. If they have not significantly changed,
# they are disregarded. 
# Weights should be recalculated whenever an occupant is updated.
# 
# The vehicle will use this information as follows:
#   - If the intended direction is into an occupied angle, review the occupant weight to determine safety
#   - If occupant weight is too high, collision avoidance mechanisms should be invoked
#       - Determine a speed adjustment and/or a steering adjustment
#   - If the occupant weight rate of change is too high, there may be an oncoming threat that should be avoided
#       - The degree of this oncoming threat will be determined by the occupant weight 

import numpy as np
from typing import Dict, List
from managers.Occupant import Occupant
import matplotlib.pyplot as plt

from uuid import uuid4

from utils.lidar_utils import distance_of_point

DISCRETIZATION_AMOUNT = 400
ANGULAR_DISCRETIZATION = (2 * np.pi) / DISCRETIZATION_AMOUNT
OCCUPANT_UPDATE_DISTANCE = 2
DEFAULT_SEARCH_RANGE = 10
EXPIRATION_PROBABILITY = 0.4


(fig, ax) = plt.subplots()

rad = 1
angle = np.linspace(0, 2 * np.pi, 150)
circx = rad * np.cos(angle)
circy = rad * np.sin(angle)

plt.ion()
plt.show()

class AngularOccupancy:
    occupancy_list: List[int]
    occupant_reference: Dict[int, Occupant]

    def __init__(self):
        self.occupancy_list = [0] * DISCRETIZATION_AMOUNT
        self.occupant_reference = {}
        
    def _occupant_search(self, centering_on, search_range = DEFAULT_SEARCH_RANGE):
        center_index = round(centering_on / (2 * np.pi) * DISCRETIZATION_AMOUNT)
        for i in range(center_index - search_range, center_index + search_range + 1):
            o = self.occupancy_list[i]
            if o != 0:
                return o
        return None

    def draw(self):
        plt.cla()
        x = []
        y = []
        for i in range(0, DISCRETIZATION_AMOUNT):
            if self.occupancy_list[i] != 0:
                occ = self.occupant_reference[self.occupancy_list[i]]
                if occ.weight >= 10:

                    angle = i / DISCRETIZATION_AMOUNT * 2.0 * np.pi
                    tx = np.cos(angle)
                    ty = np.sin(angle)
                    x.append(tx)
                    y.append(ty)
                    ax.text(tx, ty + 0.1, "{:.2f}".format(self.occupant_reference[self.occupancy_list[i]].weight))
                
        ax.set_aspect(1)
        ax.plot(circx, circy)
        ax.scatter(x, y)
        plt.draw()
        fig.canvas.flush_events()

    def expire_occupants(self):
        for i in range(0, DISCRETIZATION_AMOUNT):
            occ_ptr = self.occupancy_list[i]
            if occ_ptr != 0:
                occ = self.occupant_reference[occ_ptr]
                if occ.probability <= EXPIRATION_PROBABILITY:
                    self.occupant_reference.pop(occ_ptr).kill()
                    self.occupancy_list[i] = 0

    def occupant_from_blobs(self, blobs):
        for b in blobs:
            self.occupant_from_blob(b)

    # add, update, or ignore the new blob scan 
    def occupant_from_blob(self, new_blob):
        if len(new_blob) == 0:
            return
        start_point = new_blob[0]
        end_point = new_blob[-1]

        start_angle = np.arctan2(start_point[0], start_point[1])
        end_angle = np.arctan2(end_point[0], end_point[1])
        mid_angle = (start_angle + end_angle) / 2

        curr_occ_ptr = self._occupant_search(mid_angle)
        if curr_occ_ptr is None:
            o = Occupant(new_blob[len(new_blob) // 2])
            if o.distance > 50:
                o.kill()
                return

            new_ptr = uuid4().int
            self.occupant_reference[new_ptr] = o
        else:
            o = self.occupant_reference[curr_occ_ptr]
            if o.distance - distance_of_point(start_point, (0, 0)) > 15:
                return
            o.update(center_point=new_blob[len(new_blob) // 2])
            new_ptr = curr_occ_ptr
            # remove all angular references to previous occupant version
            for i in range(0, DISCRETIZATION_AMOUNT):
                if self.occupancy_list[i] == curr_occ_ptr:
                    self.occupancy_list[i] = 0
        # insert new pointers
        start_index = round(start_angle / (2 * np.pi) * DISCRETIZATION_AMOUNT)
        end_index = round(start_angle / (2 * np.pi) * DISCRETIZATION_AMOUNT)
        for i in range(start_index, end_index + 1): 
            self.occupancy_list[i] = new_ptr
