import numpy as np
import time

def validate_points(points: np.array) -> np.array:
    # If the user is tracking only a single point, reformat it slightly.
    if points.shape == (2,):
        points = points[np.newaxis, ...]
    elif len(points.shape) == 1:
        print("The point(s) need to be in (x, y) formats ", points)
    else:
        if points.shape[1] != 2 or len(points.shape) > 2:
            print("The point(s) need to be in (x, y) formats ", points)
    return points

class Timer():
    def __init__(self):
        pass
    
    def start(self, name: str = None):
        self._name = name
        self._startTime = time.perf_counter()
    
    def end(self, print: bool = True):
        self._timeElapsed = (time.perf_counter() - self._startTime)*1000
        if print:
            self.print()
        return self._timeElapsed
    
    def print(self):
        print(f"{self._name} took {self._timeElapsed} ms")

    @property
    def time(self):
        return self._timeElapsed