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

from threading import Lock
import numpy as np
from typing import Dict, List
from managers.Occupant import Occupant
import matplotlib.pyplot as plt

from uuid import uuid4

from utils.lidar_utils import distance_of_point

DISCRETIZATION_AMOUNT = 300
ANGULAR_DISCRETIZATION = (2 * np.pi) / DISCRETIZATION_AMOUNT
OCCUPANT_UPDATE_DISTANCE = 2
DEFAULT_SEARCH_RANGE = 2
EXPIRATION_PROBABILITY = 0.4

class AngularOccupancy:
    occupancy_list: List[int]
    occupant_reference: Dict[int, Occupant]

    def __init__(self):
        self.occupancy_list = [0] * DISCRETIZATION_AMOUNT
        self.occupant_reference = {}
        self._draw_init = False

    def _occupant_search(self, centering_on, search_range = DEFAULT_SEARCH_RANGE):
        center_index = round(centering_on / (2 * np.pi) * DISCRETIZATION_AMOUNT)
        for i in range(center_index - search_range, center_index + search_range + 1):
            o = self.occupancy_list[i]
            if o != 0:
                return o
        return None

    def draw(self):
        if not self._draw_init:
            (self.fig, self.ax) = plt.subplots()

            rad = 1
            angle = np.linspace(0, 2 * np.pi, 150)
            self.circx = rad * np.cos(angle)
            self.circy = rad * np.sin(angle)

            plt.ion()
            plt.show()
            self._draw_init = True

        plt.cla()
        x = []
        y = []
        last_label = None
        for i in range(0, DISCRETIZATION_AMOUNT):
            if self.occupancy_list[i] != 0:
                occ = self.occupant_reference[self.occupancy_list[i]]
                if occ.weight >= 10:

                    angle = i / DISCRETIZATION_AMOUNT * 2.0 * np.pi + (np.pi / 2)
                    tx = np.cos(angle)
                    ty = np.sin(angle)
                    x.append(tx)
                    y.append(ty)

                    if last_label != self.occupancy_list[i]:
                        last_label = self.occupancy_list[i]
                        self.ax.text(tx, ty + 0.1, "{:.2f}".format(self.occupant_reference[self.occupancy_list[i]].weight))
                        # self.ax.text(tx, ty + 0.1, angle)
                
        self.ax.set_aspect(1)
        self.ax.plot(self.circx, self.circy)
        self.ax.scatter(x, y)
        plt.draw()
        self.fig.canvas.flush_events()

    def expire_occupants(self):
        for i in range(0, DISCRETIZATION_AMOUNT):
            occ_ptr = self.occupancy_list[i]
            if occ_ptr != 0:
                if occ_ptr in self.occupant_reference:
                    occ = self.occupant_reference[occ_ptr]
                    if occ.probability <= EXPIRATION_PROBABILITY:
                        self.occupant_reference.pop(occ_ptr).kill()
                        self.occupancy_list[i] = 0
                else:
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

        start_angle = np.arctan2(start_point[0], start_point[1]) - (np.pi / 2)
        end_angle = np.arctan2(end_point[0], end_point[1]) - (np.pi / 2)
        mid_angle = (start_angle + end_angle) / 2

        curr_occ_ptr = self._occupant_search(mid_angle)
        new_occ = Occupant(new_blob[len(new_blob) // 2])
        new_ptr = uuid4().int
        if curr_occ_ptr is not None:
            occ = self.occupant_reference[curr_occ_ptr]
            # if big distance jump, keep closer
            if abs(occ.distance - new_occ.distance) > 15 and new_occ.distance < occ.distance:
                occ.kill()
                self.occupant_reference.pop(curr_occ_ptr)
                self.occupant_reference[new_ptr] = new_occ
            # otherwise update
            else:
                new_ptr = curr_occ_ptr
                occ.update_with(new_occ)
                new_occ.kill()
            for i in range(0, DISCRETIZATION_AMOUNT):
                    if self.occupancy_list[i] == curr_occ_ptr:
                        self.occupancy_list[i] = 0
        else:
            self.occupant_reference[new_ptr] = new_occ
        # insert new pointers
        start_index = round(start_angle / (2 * np.pi) * DISCRETIZATION_AMOUNT)
        end_index = round(end_angle / (2 * np.pi) * DISCRETIZATION_AMOUNT)
        for i in range(start_index, end_index + 1): 
            self.occupancy_list[i] = new_ptr
