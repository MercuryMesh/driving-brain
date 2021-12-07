from typing import List, TypeVar, Generic

T = TypeVar('T')
class WeightedQueue(Generic[T]):
    _queue: List[T]

    def __init__(self) -> None:
        self._queue = []

    def enqueue(self, item):
        for i, obj in enumerate(self._queue):
            if obj.weight < item.weight:
                self._queue.insert(i, obj)
                return i
        self._queue.append(obj)
        return len(self._queue) - 1

    def pop(self):
        return self._queue.pop()