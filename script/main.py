from env import Car, TRoad, Generator, Action, State
from pomcp import POMCP
import numpy as np

class Args:
    dt = 0.5
    width = 2
    length = 4

    boundary = [9.0, 4.0, -5.0, 5.0]
    road_length = 5.0
    resolution = [0.5, 0.5, 1.0, np.pi / 8]
    limits = [20.0, 20.0, 4.0, 2 * np.pi]

    maxIter = 1000

    # pomcp
    timeout = 500
    no_particles = 100

def run():
    # initialzation
    args = Args()

    cars = [Car(args.dt, args.width, args.length) for _ in range(3)]

    states = []
    states.append(State(-7.0, 4.0 + 1.25, 0.0, 0.0, 2))
    states.append(State(5 - 1.25, 2, 0.0, np.pi / 2, 0))
    states.append(State(7.0, 9.0 - 1.25, 0.0, np.pi, 1))

    for i in range(len(cars)):
        cars[i].setState(states[i])

    road = TRoad(args.boundary, args.road_length, args.resolution, args.limits)

    gens = [Generator(args.dt, args.width, args.length, len(cars), road) for _ in range(len(cars))]

    pomcps = [POMCP(gens[i], timeout=args.timeout, no_particles=args.no_particles) for i in range(len(cars))]
    
    S = []
    A = []
    O = []

    for i in range(3):
        for j in range(3):
            S.append((i, j))
    for i in range(5):
        for j in range(5):
            for k in range(5):
                a = str(i) + ' ' + str(j) + ' ' + str(k)
                A.append(a)

    for i in range(len(pomcps)):
        pomcps[i].initialize(S, A, O)

    # run env

    cur_state0 = []
    for j in range(len(cars)):
        cur_state0.append(cars[j].get_state())

    cur_state1 = [cur_state0[1], cur_state0[0], cur_state0[2]]
    cur_state2 = [cur_state0[2], cur_state0[0], cur_state0[1]]

    cur_state = [cur_state0, cur_state1, cur_state2]

    for i in range(args.maxIter):
        # search action
        
        actions = []
        for j in range(len(cars)):
            actions.append(pomcps[j].Search(cur_state[j]))

        # step
        next_state0 = []
        next_obs = []
        real_act = []
        for j in range(len(cars)):
            # extract action
            _act = int(actions[j].split(' ')[0])

            print("j: ", actions[j])
            real_act.append(_act)
            cars[j].update(_act)
            next_state0.append(cars[j].get_state())
        
        next_state1 = [next_state0[1], next_state0[0], next_state0[2]]
        next_state2 = [next_state0[2], next_state0[0], next_state0[1]]

        next_state = [next_state0, next_state1, next_state2]

        for j in range(len(cars)):
            next_obs.append(road.get_index(next_state[j]))

        # update belief
        act_str = []
        act_str.append(str(real_act[0]) + ' ' + str(real_act[1]) + ' ' + str(real_act[2]))
        act_str.append(str(real_act[1]) + ' ' + str(real_act[0]) + ' ' + str(real_act[2]))
        act_str.append(str(real_act[2]) + ' ' + str(real_act[0]) + ' ' + str(real_act[1]))
        for j in range(len(pomcps)):
            pomcps[j].UpdateBelief(act_str[j], next_obs[j])
            pomcps[j].tree.prune_after_action(act_str[j], next_obs[j])

        cur_state = next_state

        print(cur_state[0][0].get_state())
        print(cur_state[0][1].get_state())
        print(cur_state[0][2].get_state())
        # print("car0: [", cur_state[0][1].x, ", ", cur_state[0][1].y, "]")
        # print("car0: [", cur_state[0][2].x, ", ", cur_state[0][2].y, "]")

run()
