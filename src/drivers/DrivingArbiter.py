from airsim.client import CarClient
from typing import Callable, List, NamedTuple
from threading import Event, Thread, Timer
from drivers.Driver import Driver, DriverPriority
from utils import airsimIOLock, GracefulKiller

class Requester(NamedTuple):
    id: str
    onGranted: Callable
    onRevoked: Callable

class Allocation:
    driver: Requester
    priority: DriverPriority

    def __init__(self, driver, priority):
        self.driver = driver
        self.priority = priority

RequestQueue = List[Allocation]

class Arbiter:
    _requestQueue: RequestQueue = []
    _currentAllocation: Allocation = None

    # After a Driver receives an allocation, it may return control to the previous Driver
    # This should be specified in the requestControl function
    _returnDriver: Allocation = None

    # insert ahead of the first object you have priority over
    def _findInsertIndex(self, priority: DriverPriority) -> int:        
        for i, val in enumerate(self._requestQueue):
            if priority.hasPriority(val):
                return i
        return len(self._requestQueue)

    def requestControl(self, requester: Requester, priority: DriverPriority, returnControl=True) -> bool:
        if self._currentAllocation is None:
            requester.onGranted()
            return True

        # if they already have control, don't change grant
        if (requester.id == self._currentAllocation.driver.id):
            requester.onGranted()
            return True

        if DriverPriority.hasPriority(requester, self._currentAllocation.priority):
            if returnControl:
                self._returnDriver = self._currentAllocation
            # revoke control
            self._currentAllocation.driver.onRevoked(returnControl)
            # grant control
            requester.onGranted()
            self._currentAllocation = Allocation(requester, priority)
            return True
        else:
            index = self._findInsertIndex(priority)
            self._requestQueue.insert(index, Allocation(requester, priority))
            return False
    
    def giveUpControl(self, requester: Requester):
        # nop if not currently in control
        if requester.id != self._currentAllocation.driver.id:
            return

        requester.onRevoked(False)
        
        if self._returnDriver is not None:
            self._currentAllocation = self._returnDriver
            self._currentAllocation.driver.onGranted()
            self._returnDriver = None
            return

        if len(self._requestQueue) == 0:
            raise RuntimeError("giveUpControl called with no elements in the request queue")
        
        self._currentAllocation = self._requestQueue.pop(0)
        self._currentAllocation.driver.onGranted()

class DrivingArbiter:
    _speedArbiter = Arbiter()
    _steeringArbiter = Arbiter()

    def __init__(self, client: CarClient):
        self._client = client
        self._speedController = SpeedController(client)
        self._steeringController = SteeringController(client)
        # GracefulKiller(self._updating.clear)

    def requestSteeringControl(self, driver: Driver, priority: DriverPriority, returnControl=True):
        requester = Requester(driver.id, lambda: driver.onSteeringGranted(self._steeringController), driver.onSteeringRevoked)
        return self._steeringArbiter.requestControl(requester, priority, returnControl)

    def requestSpeedControl(self, driver: Driver, priority: DriverPriority, returnControl=True):
        requester = Requester(driver.id, lambda: driver.onSpeedGranted(self._speedController), driver.onSpeedRevoked)
        return self._speedArbiter.requestControl(requester, priority, returnControl)

    def giveUpSteeringControl(self, driver: Driver):
        requester = Requester(driver.id, lambda: driver.onSteeringGranted(self._steeringController), driver.onSteeringRevoked)
        return self._steeringArbiter.giveUpControl(requester)

    def giveUpSpeedControl(self, driver: Driver):
        requester = Requester(driver.id, lambda: driver.onSpeedGranted(self._speedController), driver.onSpeedRevoked)
        return self._speedArbiter.giveUpControl(requester)

    def sendBatch(self):
        controls = self._client.getCarControls()
        controls.throttle = self._speedController.throttle
        controls.brake = self._speedController.brake
        controls.steering = self._steeringController.steering
        self._client.setCarControls(controls)

class SpeedController:
    def __init__(self, client: CarClient):
       controls = client.getCarControls()
       self.throttle = controls.throttle
       self.brake = controls.brake

class SteeringController:
    steering = 0

    def __init__(self, client: CarClient):
        carControls = client.getCarControls()
        self.steering = carControls.steering
