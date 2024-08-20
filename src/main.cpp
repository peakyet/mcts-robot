
#include <iostream>
#include <string>
#include <cmath>
#include <vector>

#include <iomanip>  // for std::setw and std::setfill

#include "pomcp.hpp"
#include "env.hpp"

#include <matplotlib-cpp/matplotlibcpp.h>
namespace plt = matplotlibcpp;

struct Args {
    double dt = 0.5;
    double width = 2.0;
    double length = 4.0;

    std::vector<double> boundary = {8.0, 2.0, -3.0, 3.0};
    double road_length = boundary[0] - boundary[1];
    std::vector<double> resolution = {0.25, 0.25, 0.5, M_PI / 16};
    std::vector<double> limits = {20.0, 20.0, 6.0, 2 * M_PI};

    int maxIter = 80;

    double gamma = 0.95;
    double c = 5;
    double threshold = 0.7;
    int timeout = 3000;
    int no_particles = 1000;
};

void run() {
    // Initialization
    using namespace std;
    cout << "init" << endl;
    std::srand(std::time(nullptr));
    Args args;

    std::vector<Car> cars;
    for (int i = 0; i < 2; ++i) {
        cars.emplace_back(args.dt, args.width, args.length);
    }

    std::vector<State> states = {
        State(-10.0, args.boundary[1] + args.road_length / 2, 0.0, 0.0, 2),
        State(args.boundary[3] - args.road_length / 2, -6, 0.0, M_PI / 2, 0),
        State(10, args.boundary[0] - args.road_length / 4, 0.0, M_PI, 1)
    };

    for (size_t i = 0; i < cars.size(); ++i) {
        cars[i].setState(states[i]);
    }

    TRoad road(args.boundary, args.road_length, args.resolution, args.limits);

    std::vector<Generator> gens;
    for (int i = 0; i < cars.size(); ++i) {
        gens.emplace_back(args.dt, args.width, args.length, cars.size(), road);
    }

    std::vector<POMCP> pomcps;
    for (int i = 0; i < cars.size(); ++i) {
        pomcps.emplace_back(&gens[i], args.gamma, args.c, args.threshold, args.timeout, args.no_particles);
    }

    std::vector<std::vector<int>> S;
    std::vector<std::string> A;
    std::vector<std::string> O;

    for (int i = 0; i < 3; ++i) {
        S.push_back({i});
        // for (int j = 0; j < 3; ++j) {
        //     S.push_back({i, j});
        // }
    }

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            A.emplace_back(std::to_string(i) + " " + std::to_string(j));
            // for (int k = 0; k < 5; ++k) {
            //     A.emplace_back(std::to_string(i) + " " + std::to_string(j) + " " + std::to_string(k));
            // }
        }
    }

    for (int i = 0; i < pomcps.size(); i++) {
        pomcps[i].initialize(S, A, O);
    }

    // Run environment
    // std::vector<std::vector<State>> cur_state = {
    //     { cars[0].get_state(), cars[1].get_state(), cars[2].get_state() },
    //     { cars[1].get_state(), cars[0].get_state(), cars[2].get_state() },
    //     { cars[2].get_state(), cars[0].get_state(), cars[1].get_state() }
    // };
    std::vector<std::vector<State>> cur_state = {
        { cars[0].get_state(), cars[1].get_state()},
        { cars[1].get_state(), cars[0].get_state()},
    };

    cout << "run" << endl;

    for (int i = 0; i < args.maxIter; ++i) {
        // Search action
        std::vector<int> actions;
        for (int j = 0; j < cars.size(); ++j) {
            actions.push_back(pomcps[j].Search(cur_state[j]));
        }
        // actions[0] = 1;

        // Step
        std::vector<State> next_state0;
        std::vector<std::string> next_obs;

        for (int j = 0; j < cars.size(); ++j) {
            cars[j].update(actions[j]);
            next_state0.push_back(cars[j].get_state());
        }

        // std::vector<std::vector<State>> next_state = {
        //     { next_state0[0], next_state0[1], next_state0[2] },
        //     { next_state0[1], next_state0[0], next_state0[2] },
        //     { next_state0[2], next_state0[0], next_state0[1] }
        // };
        std::vector<std::vector<State>> next_state = {
            { cars[0].get_state(), cars[1].get_state()},
            { cars[1].get_state(), cars[0].get_state()},
        };

        for (int j = 0; j < next_state.size(); ++j) {
            next_obs.push_back(road.get_index(next_state[j]));
        }

        // Update belief
        // std::vector<std::string> act_str = {
        //     std::to_string(actions[0]) + " " + std::to_string(actions[1]) + " " + std::to_string(actions[2]),
        //     std::to_string(actions[1]) + " " + std::to_string(actions[0]) + " " + std::to_string(actions[2]),
        //     std::to_string(actions[2]) + " " + std::to_string(actions[0]) + " " + std::to_string(actions[1])
        // };
        std::vector<std::string> act_str = {
            std::to_string(actions[0]) + " " + std::to_string(actions[1]),
            std::to_string(actions[1]) + " " + std::to_string(actions[0]),
        };

        for (int j = 0; j < pomcps.size(); ++j) {
            pomcps[j].UpdateBelief(act_str[j], next_obs[j]);
        }

        cur_state = next_state;
       
        // Plotting code
        plt::cla();
        road.draw();

        for (int j = 0; j < 3; j++){
            auto goal = road.get_goal(j);
            plt::plot({goal[0]}, {goal[1]}, {{"marker", "x"}, {"color", "green"}});
        }

        for (int j = 0; j < cars.size(); ++j) {
            auto car_state = cars[j].get_state();
            auto goal = road.get_goal(car_state.g);

            // Example drawing for each car
            // plt::plot({car_state.x}, {car_state.y}, {{"marker", "o"}, {"color", "blue"}});
            cars[j].draw();
            plt::text(car_state.x, car_state.y, "Robot");

            std::cout << "x: " << car_state.x << ", y: " << car_state.y << ", theta: "<< car_state.theta<< ", g: " << car_state.g << std::endl;
        }

        plt::xlim(-10, 10);
        plt::ylim(-10, 10);
        plt::set_aspect_equal();

        // 保存帧为PNG文件
        // std::stringstream ss;
        // ss << "frame_" << std::setw(4) << std::setfill('0') << i << ".png";
        // plt::save(ss.str());        

        plt::pause(0.01);  // Adjust as needed for simulation speed

        // plt::show();


    }
}

int main() {
    run();
    return 0;
}
