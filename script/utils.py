
class State:
    def __init__(self, x=0, y=0, v=0, theta=0, g=0):
        self.x = x
        self.y = y
        self.v = v
        self.theta = theta
        self.g = g

from enum import Enum

class Action(Enum):
    STOP = 0
    ACC = 1
    DEACC = 2
    LEFT = 3
    RIGHT = 4
