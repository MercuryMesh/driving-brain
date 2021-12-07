from typing import Tuple
from threading import Event, Thread
from time import sleep


class RerunableThread:
    args: Tuple = None

    @property
    def is_running(self) -> bool:
        return self._is_running.is_set()

    def __init__(self, target, name=None):
        self._is_running = Event()
        self._target = target
        self._thread = Thread(target=self._threadRunLoop, name=name)
        self._thread.setDaemon(True)
        self._thread.start()

    def run(self, args: Tuple):
        # print(args)
        self.args = args
        self._is_running.set()

    def _threadRunLoop(self):
        while True:
            if self._is_running.is_set():
                self._target(*(self.args))
                self._is_running.clear()
            else:
                sleep(0.01)
