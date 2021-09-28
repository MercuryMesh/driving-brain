from abc import ABC, abstractmethod
import enum

from drivers.DrivingArbiter import SpeedController, SteeringController

class RequestType(enum):
    Steering = 1
    Speed = 2
    All = 3

class DriverPriority(enum):
    # only interrupts if has a strictly higher priority
    Low = 0
    Medium = 1
    High = 2

    # inuterrupts everything but uninterruptable
    Crucial = 3

    # always interrupts and cannot be interrupted
    Uninteruptable = 100

    @staticmethod
    def hasPriority(this, other):
        if other == DriverPriority.Uninteruptable:
            return False
        if other == DriverPriority.Crucial:
            return this >= other
        return this > other


class Driver(ABC):

    @abstractmethod
    @property
    def id() -> str:
        pass

    def equals(self, other) -> bool:
        return self.id == other.id

    @abstractmethod
    def onSteeringGranted(self, controller: SteeringController):
        pass

    @abstractmethod
    def onSpeedGranted(self, controller: SpeedController):
        pass

    @abstractmethod
    def onSteeringRevoked(self):
        pass

    @abstractmethod
    def onSpeedRevoked(self):
        pass