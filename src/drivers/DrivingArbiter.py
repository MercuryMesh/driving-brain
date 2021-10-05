from airsim.client import CarClient
from typing import Callable, List, NamedTuple
from threading import Event, Lock, Thread, Timer
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

# Arbiters should be thread-safe by using locks to protect shared memory accesses.
class Arbiter:
    _queueLock: Lock = None
    _requestQueue: RequestQueue = None
    _currentAllocation: Allocation = None

    # After a Driver receives an allocation, it may return control to the previous Driver
    # This should be specified in the requestControl function
    _returnDriver: Allocation = None

    def __init__(self):
        self._requestQueue = []
        self._queueLock = Lock()

    # insert ahead of the first object you have priority over
    def _findInsertIndex(self, priority: DriverPriority) -> int:        
        for i, val in enumerate(self._requestQueue):
            if priority.hasPriority(val):
                return i
        return len(self._requestQueue)

    def requestControl(self, requester: Requester, priority: DriverPriority, returnControl=True) -> bool:
        self._queueLock.acquire()
        if self._currentAllocation is None:
            self._queueLock.release()
            requester.onGranted()
            return True

        # if they already have control, don't change grant
        if (requester.id == self._currentAllocation.driver.id):
            self._queueLock.release()
            requester.onGranted()
            return True

        if DriverPriority.hasPriority(requester, self._currentAllocation.priority):
            if returnControl:
                self._returnDriver = self._currentAllocation
            self._currentAllocation = Allocation(requester, priority)
            self._queueLock.release()

            # revoke control
            self._currentAllocation.driver.onRevoked(returnControl)
            # grant control
            requester.onGranted()
            return True
        else:
            index = self._findInsertIndex(priority)
            self._requestQueue.insert(index, Allocation(requester, priority))
            self._queueLock.release()
            return False
    
    def giveUpControl(self, requester: Requester):
        self._queueLock.acquire()
        # nop if not currently in control
        if requester.id != self._currentAllocation.driver.id:
            self._queueLock.release()
            return

        requester.onRevoked(False)
        
        if self._returnDriver is not None:
            self._currentAllocation = self._returnDriver
            self._returnDriver = None
            self._queueLock.release()
            self._currentAllocation.driver.onGranted()
            return

        if len(self._requestQueue) == 0:
            raise RuntimeError("giveUpControl called with no elements in the request queue")
        
        self._currentAllocation = self._requestQueue.pop(0)
        self._queueLock.release()
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
