import math
from enum import Enum
import numpy as np
import copy

class State:
    def __init__(self, x=0.0, y=0.0, v=0.0, theta=0.0, g=0):
        self.x = x
        self.y = y
        self.v = v
        self.theta = theta
        self.g = g # goal 0, 1, 2

    def get_state(self):
        return [self.x, self.y, self.v, self.theta, self.g]

class Action():
    STOP = 0
    ACC = 1
    DEACC = 2
    LEFT = 3
    RIGHT = 4

class Car:
    def __init__(self, dt, width, length):
        self.s0 = State()
        self.dt = dt
        self.width = width
        self.length = length
        self.action = Action()

    def setState(self, state: State):
        self.s0.x = state.x
        self.s0.y = state.y
        self.s0.v = state.v
        self.s0.theta = state.theta
        self.s0.g = state.g
    
    def get_state(self):
        return copy.deepcopy(self.s0)

    def update(self, act):
        _act = [0, 0]  # [acceleration, angular velocity]
        if act == self.action.STOP:
            _act = [0, 0]
        elif act == self.action.ACC:
            _act = [2.0, 0]
        elif act == self.action.DEACC:
            _act = [-2.0, 0]
        elif act == self.action.LEFT:
            _act = [0, -np.pi / 2]
        elif act == self.action.RIGHT:
            _act = [0, np.pi / 2]
        else:
            return False

        self.s0.x += self.s0.v * math.cos(self.s0.theta) * self.dt
        self.s0.y += self.s0.v * math.sin(self.s0.theta) * self.dt
        self.s0.v += _act[0] * self.dt
        self.s0.theta += _act[1] * self.dt


        if self.s0.theta > 2 * np.pi:
            self.s0.theta -= 2 * np.pi
        if self.s0.theta < 0:
            self.s0.theta += 2 * np.pi

        return True

    def check_collision(self, car1):
        s1 = car1.get_state()
        
        # AABB (Axis-Aligned Bounding Box) collision detection
        dx = abs(self.s0.x - s1.x)
        dy = abs(self.s0.y - s1.y)

        return (dx < self.length and dy < self.width)

    def set_intention(self, g):
        self.s0.g = g
        return True

class TRoad:
    def __init__(self, boundary, road_length, resolution, limits):
        self.boundary = {
            'up': boundary[0],
            'down': boundary[1],
            'left': boundary[2],
            'right': boundary[3]
        }
        self.road_length = road_length

        self.x_res = resolution[0]
        self.y_res = resolution[1]
        self.v_res = resolution[2]
        self.t_res = resolution[3]

        self.x_max = limits[0]
        self.y_max = limits[1]
        self.v_max = limits[2]
        self.t_max = limits[3]

        self.xCell_max = int(self.x_max / self.x_res)
        self.yCell_max = int(self.y_max / self.y_res)
        self.vCell_max = int(self.v_max / self.v_res)
        self.tCell_max = int(self.t_max / self.t_res)

        self.increase = self.xCell_max * self.yCell_max * self.vCell_max * self.tCell_max

    def out_road(self, car0):
        s0 = car0.get_state()

        if (s0.y > self.boundary['up'] or
            (s0.x < self.boundary['left'] and s0.y < self.boundary['down']) or
            (s0.x > self.boundary['right'] and s0.y < self.boundary['down'])):
            return True
        else:
            return False

    def get_goal(self, goal):
        if goal == 0:
            return np.array([-10, self.boundary['up'] - self.road_length / 4.0])
        elif goal == 1:
            return np.array([self.boundary['left'] + self.road_length / 4.0, -10])
        elif goal == 2:
            return np.array([10, self.boundary['up'] - self.road_length / 4.0])
        else:
            return np.array([0, 0])

    def get_index(self, cars_state):
        index = ''
        for s0 in cars_state:
            ind = 0
            # s0 = car.get_state()

            x = np.clip(s0.x, -self.x_max / 2.0, self.x_max / 2.0) + self.x_max / 2.0
            y = np.clip(s0.y, -self.y_max / 2.0, self.y_max / 2.0) + self.y_max / 2.0
            v = np.clip(s0.v, -self.v_max / 2.0, self.v_max / 2.0) + self.v_max / 2.0
            theta = np.clip(s0.theta, -self.t_max / 2.0, self.t_max / 2.0) + self.t_max / 2.0

            x_cell = int(x / self.x_res)
            y_cell = int(y / self.y_res)
            v_cell = int(v / self.v_res)
            t_cell = int(theta / self.t_res)

            ind += x_cell + y_cell*self.xCell_max + v_cell * self.xCell_max * self.yCell_max + t_cell * self.xCell_max * self.yCell_max * self.vCell_max

            index = index + str(ind) + ' '

        index = index[:-1]

        return index

    def get_state(self, index):
        cars_index = index.split(' ')

        states = []

        for car_index in cars_index:
            ind = int(car_index)
            s = State()

            x_cell = ind % self.xCell_max
            ind = int(ind / self.xCell_max)
            s.x = x_cell * self.x_res
            
            y_cell = ind % self.yCell_max
            ind = int(ind / self.yCell_max)
            s.y = y_cell * self.y_res

            v_cell = ind % self.vCell_max
            ind = int(ind / self.vCell_max)
            s.v = v_cell * self.v_res

            t_cell = ind
            s.t = t_cell * self.t_res

            states.append(s)

        return states
        
class Generator:
    def __init__(self, dt, width, length, num, road_env):
        self.sim_cars = [Car(dt, width, length) for _ in range(num)]
        self.road_env = road_env

    def gen(self, states: list, action: str):

        cars_state = []
        act = []
        act_tmp = action.split(' ')
        for ac in act_tmp:
            act.append(int(ac))
            
        
        for i in range(len(self.sim_cars)):
            self.sim_cars[i].setState(states[i])
            self.sim_cars[i].update(act[i])

            cars_state.append(self.sim_cars[i].get_state())

        obs_index = self.road_env.get_index(cars_state)

        reward = self.get_reward(act, self.road_env)

        return [cars_state,obs_index, reward]
      

    def get_reward(self, act: list, road_env: TRoad):
        # r_goal + r_collision + r_acc + r_road + r_opp

        reward = 0.0

        car_num = len(self.sim_cars)
        for i in range(car_num):
            s0 = self.sim_cars[i].get_state()
            goal = road_env.get_goal(s0.g)
            cur_pos = np.array([s0.x, s0.y])

            # goal
            reward -= 0.5 * np.linalg.norm(cur_pos - goal)

            # collision
            col = False
            if (i + 1 <= car_num - 1):
                for j in range(i+1, car_num):
                    if (self.sim_cars[i].check_collision(self.sim_cars[j])):
                        col = True

            if col:
                reward -= 50

            # acc
            if act[i] != 0:
                reward -= 5

            # road
            if (road_env.out_road(self.sim_cars[i])):
                reward -= 50

        return reward
