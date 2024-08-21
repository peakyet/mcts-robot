
#include <iostream>
#include <string>
#include <cmath>
#include <vector>

#include <iomanip>  // for std::setw and std::setfill

#include "mcts.hpp"
#include "env.hpp"

#include <matplotlib-cpp/matplotlibcpp.h>
namespace plt = matplotlibcpp;

struct Args {
    double dt = 0.5;
    double width = 2.0;
    double length = 4.0;

    std::vector<double> boundary = {8.0, 0.0, -4.0, 4.0};
    double road_length = boundary[0] - boundary[1];
    std::vector<double> resolution = {0.25, 0.25, 0.5, M_PI / 16};
    std::vector<double> limits = {20.0, 20.0, 6.0, 2 * M_PI};

    int maxIter = 80;

    double gamma = 0.95;
    double c = 500;
    double threshold = 0.6;
    int timeout = 10000;
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
        State(-10.0, args.boundary[1] + args.road_length / 2, 0.0, 0.0, 1),
        State(args.boundary[3] - args.road_length / 2, -6, 0.0, M_PI / 2, 0),
        State(10, args.boundary[0] - args.road_length / 4, 0.0, M_PI, 0)
    };

    for (size_t i = 0; i < cars.size(); ++i) {
        cars[i].setState(states[i]);
    }

    TRoad road(args.boundary, args.road_length, args.resolution, args.limits);

    std::vector<Generator> gens;
    for (int i = 0; i < cars.size(); ++i) {
        gens.emplace_back(args.dt, args.width, args.length, cars.size(), road);
    }

    std::vector<MCTS> mcts;
    for (int i = 0; i < cars.size(); ++i) {
        mcts.emplace_back(&gens[i], args.gamma, args.c, args.threshold, args.timeout, args.no_particles);
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

    for (int i = 0; i < mcts.size(); i++) {
        mcts[i].initialize(S, A, O);
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
    // plt::figure();
    plt::figure_size(1600, 1080);  // 单位为英寸
    for (int i = 0; i < args.maxIter; ++i) {
        // Search action
        std::vector<int> actions;
        for (int j = 0; j < cars.size(); ++j) {
            actions.push_back(mcts[j].Search(cur_state[j]));
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

        std::vector<std::vector<double>> Pr;
        for (int j = 0; j < mcts.size(); ++j) {
            Pr.push_back(mcts[j].UpdateBelief(act_str[j], next_obs[j]));
        }

        cur_state = next_state;
       
        // Plotting code
        plt::cla();
        plt::subplot2grid(2,3, 0, 0, 2, 2 );
        road.draw();

        for (int j = 0; j < 3; j++){
            auto goal = road.get_goal(j);
            // plt::plot({goal[0]}, {goal[1]}, {{"marker", std::to_string(j)}, {"color", "green"}});
            plt::text(goal[0], goal[1], std::to_string(j), {{"color", "green"}, {"fontsize", "20"}});
        }

        for (int j = 0; j < cars.size(); ++j) {
            auto car_state = cars[j].get_state();
            auto goal = road.get_goal(car_state.g);

            // Example drawing for each car
            // plt::plot({car_state.x}, {car_state.y}, {{"marker", "o"}, {"color", "blue"}});
            cars[j].draw();
            plt::text(car_state.x, car_state.y, "Robot_" + std::to_string(j));

            std::cout << "x: " << car_state.x << ", y: " << car_state.y << ", theta: "<< car_state.theta<< ", g: " << car_state.g << std::endl;
        }

        plt::xlim(-10, 10);
        plt::ylim(-10, 10);
        plt::title("Simulation");
        plt::xlabel("x");
        plt::xlabel("y");
        plt::set_aspect_equal();

        // plt::subplot2grid({2,3},{0,2});
        std::vector<int> intentions = {0, 1, 2};
        plt::subplot2grid(2,3,0,2);
        plt::bar(Pr[0]);
        plt::title("Probability of Robot_1's intention");
        plt::xlim(-1,3);
        plt::ylim(0, 1);
        plt::xticks(intentions);
        plt::xlabel("Intention");
        plt::ylabel("Pr");

        plt::subplot2grid(2,3,1,2);
        plt::title("Probability of Robot_0's intention");
        plt::bar(Pr[1]);
        plt::xlim(-1,3);
        plt::ylim(0, 1);
        plt::xticks(intentions);
        plt::xlabel("Intention");
        plt::ylabel("Pr");
        

        // 保存帧为PNG文件
        std::stringstream ss;
        ss << "frame_" << std::setw(4) << std::setfill('0') << i << ".png";
        plt::save(ss.str());        

        plt::pause(0.01);  // Adjust as needed for simulation speed

        // plt::show();


    }
}

int main() {
    run();
    return 0;
}
