import enum
from daq.LidarDelegate import isInFront
from drivers.Driver import Driver, DriverPriority
from drivers.DrivingArbiter import DrivingArbiter, SpeedController, SteeringController
from managers.AngularOccupancy import DISCRETIZATION_AMOUNT, AngularOccupancy
import numpy as np

ACTIVATION_WEIGHT = 10
FRONT_ANGLE = np.pi / 16

class CollisionStrategy(enum.Enum):
    none = -1
    braking = 0
    left = 1
    right = 2

class SwerveDirection(int):
    left = -1
    right = 1

class CollisionWatchdog(Driver):
    id = "collision-watchdog"
    _speedController: SpeedController
    _steeringController: SteeringController
    _collisionStrategy: CollisionStrategy
    _safeContinueCount: int
    _swerveCount: int

    def __init__(self, drivingArbiter: DrivingArbiter, angularOccupancy: AngularOccupancy):
        self._drivingArbiter = drivingArbiter
        self._angularOccupancy = angularOccupancy
        self._collisionStrategy = CollisionStrategy.none
        self._safeContinueCount = 0
        self._swerveCount = 0

    def runLoop(self, currentSpeed: float):
        if self._collisionStrategy == CollisionStrategy.braking:
            self.avoidViaBraking()
            return
        elif self._collisionStrategy == CollisionStrategy.left:
            self.swerveLeft(currentSpeed)
            
        elif self._collisionStrategy == CollisionStrategy.right:
            self.swerveRight(currentSpeed)
            
        
        for i, occptr in enumerate(self._angularOccupancy.occupancy_list):
            if occptr == 0:
                continue
            
            occ = self._angularOccupancy.occupant_reference[occptr]
            if occ.weight < ACTIVATION_WEIGHT:
                continue
            
            angle = (i / DISCRETIZATION_AMOUNT) * np.pi * 2

            if (abs(angle) < FRONT_ANGLE) and occ.center_point[0] > 0:
                # print("brake")
                # print(f"for weight {occ.weight} and angle {angle}")
                # print(occ.center_point)
                self._collisionStrategy = CollisionStrategy.braking
                self._drivingArbiter.requestSpeedControl(self, DriverPriority.High)
                return
            elif angle < (np.pi) and self._collisionStrategy != CollisionStrategy.right:
                # print("swerve right")
                # print(f"for weight {occ.weight} and angle {angle}")
                # print(occ.center_point)
                # print(occ.relative_velocity)
                self._collisionStrategy = CollisionStrategy.right
                self._drivingArbiter.requestSteeringControl(self, DriverPriority.High)
                self._drivingArbiter.requestSpeedControl(self, DriverPriority.Medium)
                return
            elif angle > (np.pi) and self._collisionStrategy != CollisionStrategy.left:
                # print("swerve left")
                # print(f"for weight {occ.weight} and angle {angle}")
                # print(occ.center_point)
                # print(occ.relative_velocity)
                self._collisionStrategy = CollisionStrategy.left
                self._drivingArbiter.requestSteeringControl(self, DriverPriority.High)
                self._drivingArbiter.requestSpeedControl(self, DriverPriority.Medium)
                return

    def swerve(self, currentSpeed: float, dir: SwerveDirection):
        if currentSpeed > 8:
            self._speedController.throttle = 0
            self._speedController.brake = 3
        else:
            self._speedController.throttle = 1
            self._speedController.brake = 0
        
        rightFree = True
        for i in range(0, dir * DISCRETIZATION_AMOUNT // 2):
            occptr = self._angularOccupancy.occupancy_list[i]
            if occptr != 0:
                occ = self._angularOccupancy.occupant_reference[occptr]
                if occ.distance <= 30:
                    self._swerveCount = self._swerveCount + dir
                    self._steeringController.set_steering(dir * np.pi / 8)
                    rightFree = False
                    break
                elif occ.distance <= 20:
                    self._swerveCount = self._swerveCount + (dir * 2)
                    self._steeringController.set_steering(dir * np.pi / 4)
                    rightFree = False
                    break
                else:
                    self._steeringController.set_steering(dir * np.pi / 32)

        if rightFree:
            self._safeContinueCount += 1
            if self._safeContinueCount >= 5:
                # print("correcting")
                self._swerveCount = self._swerveCount - dir
                self._steeringController.set_steering(-dir * np.pi / 8)
                if self._swerveCount >= 0:
                    # print("return")
                    self._swerveCount = 0
                    self._safeContinueCount = 0
                    self._drivingArbiter.giveUpSteeringControl(self)
                    self._drivingArbiter.giveUpSpeedControl(self)
        else:
            self._safeContinueCount = 0
        


    def swerveLeft(self, currentSpeed: float):
        return self.swerve(currentSpeed, SwerveDirection.left)
    
    def swerveRight(self, currentSpeed: float):
        return self.swerve(currentSpeed, SwerveDirection.right)

    def avoidViaBraking(self):
        self._speedController.throttle = 0
        self._speedController.brake = 100

        index = round((FRONT_ANGLE / (2 * np.pi)) * DISCRETIZATION_AMOUNT)
        safe_to_continue = True
        for i in range(-index, index + 1):
            occptr = self._angularOccupancy.occupancy_list[i]
            if occptr != 0:
                occ = self._angularOccupancy.occupant_reference[occptr]
                safe_to_continue = safe_to_continue and (occ.distance > 30) and (occ.weight < ACTIVATION_WEIGHT)
        if safe_to_continue:
            if self._safeContinueCount >= 25:
                # print("return")
                self._drivingArbiter.giveUpSpeedControl(self)
                self._safeContinueCount = 0
            self._safeContinueCount = self._safeContinueCount + 1
        else:
            self._safeContinueCount = 0

    def onSpeedRevoked(self, willReturn):
        self._speedController = None
        self._collisionStrategy = CollisionStrategy.none
    def onSpeedGranted(self, controller):
        self._speedController = controller

    def onSteeringGranted(self, controller):
        self._steeringController = controller

    def onSteeringRevoked(self, willReturn):
        self._steeringController = None
        self._collisionStrategy = CollisionStrategy.none

    
