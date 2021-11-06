from abc import ABC, abstractmethod
from enum import Enum

class RequestType(Enum):
    Steering = 1
    Speed = 2
    All = 3

class DriverPriority(Enum):
    # only interrupts if has a strictly higher priority
    Low = 0
    Medium = 1
    High = 2

    # inuterrupts everything but uninterruptable
    Crucial = 3

    # always interrupts and cannot be interrupted
    Uninteruptable = 100

    @staticmethod
    def hasPriority(this: Enum, other: Enum):
        if other == DriverPriority.Uninteruptable:
            return False
        if other == DriverPriority.Crucial:
            return this.value >= other.value
        return this.value > other.value


class Driver(ABC):
    @property
    @abstractmethod
    def id(self):
        pass

    def equals(self, other) -> bool:
        return self.id == other.id

    @abstractmethod
    def onSteeringGranted(self, controller):
        pass

    @abstractmethod
    def onSpeedGranted(self, controller):
        pass

    @abstractmethod
    def onSteeringRevoked(self, willReturn):
        pass

    @abstractmethod
    def onSpeedRevoked(self, willReturn):
        pass