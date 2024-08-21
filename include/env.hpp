#ifndef ENV_HPP_
#define ENV_HPP_

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <limits>

#include <matplotlib-cpp/matplotlibcpp.h>
namespace plt = matplotlibcpp;

class State {
public:
    double x, y, v, theta;
    int g; // goal 0, 1, 2

    State(double x = 0.0, double y = 0.0, double v = 0.0, double theta = 0.0, int g = 0)
        : x(x), y(y), v(v), theta(theta), g(g) {}

    std::vector<double> get_state() const {
        return {x, y, v, theta, static_cast<double>(g)};
    }
};

enum Action {
    STOP = 0,
    ACC = 1,
    DEACC = 2,
    LEFT = 3,
    RIGHT = 4
};

class Car {
public:
    double dt;
    double width;
    double length;
    State s0;

    Car(double dt = 0.0, double width = 0.0, double length = 0.0)
        : dt(dt), width(width), length(length) {}

    void setState(const State& state) {
        s0 = state;
    }

    State get_state() const {
        return s0;
    }

    bool update(int act) {
        return step(s0, act);
    }

    bool step(State & s, const int &act){
        double _act[2] = {0, 0}; // [acceleration, angular velocity]

        switch (act) {
            case STOP:
                _act[0] = 0;
                _act[1] = 0;
                break;
            case ACC:
                _act[0] = 2.0;
                _act[1] = 0;
                break;
            case DEACC:
                _act[0] = -2.0;
                _act[1] = 0;
                break;
            case LEFT:
                _act[0] = 0;
                _act[1] = -M_PI / 4;
                break;
            case RIGHT:
                _act[0] = 0;
                _act[1] = M_PI / 4;
                break;
            default:
                return false;
        }

        s.x += s.v * cos(s.theta) * dt;
        s.y += s.v * sin(s.theta) * dt;
        s.v += _act[0] * dt;
        s.theta += _act[1] * dt;

        // Normalize angle
        if (s.theta > 2 * M_PI) s.theta -= 2 * M_PI;
        if (s.theta < 0) s.theta += 2 * M_PI;
        // if (s.v < 0) s.v = 0;
        if (s.v < -2) s.v = -2;
        if (s.v > 2) s.v = 2;

        return true;
    }

    bool check_collision(const Car& other) const {
        const State& s1 = other.get_state();

        // AABB (Axis-Aligned Bounding Box) collision detection
        double dx = fabs(s0.x - s1.x);
        double dy = fabs(s0.y - s1.y);

        std::cout << "x: " << s0.x << ",y: " << s0.y << std::endl;
        std::cout << "x1: " << s1.x << ",y1: " << s1.y << std::endl;
        std::cout << "dx: " << dx << ",dy: " << dy << std::endl;

        return (dx < length && dy < width);
    }

    bool set_intention(int g) {
        s0.g = g;
        return true;
    }

    void draw(){
        std::vector<std::vector<double>> points = {{length / 2.0, width / 2.0}, {-length / 2.0, width / 2.0},
                                                    {-length / 2.0, -width / 2.0}, {length / 2.0, -width / 2.0}};
        for (int i = 0; i < 4; i++){
            auto ne_x = s0.x + cos(s0.theta) * points[i][0] - sin(s0.theta) * points[i][1];
            points[i][1] = s0.y + sin(s0.theta) * points[i][0] + cos(s0.theta) * points[i][1];
            points[i][0] = ne_x;
        }
        plt::plot({points[0][0], points[1][0], points[2][0], points[3][0], points[0][0]},
                    {points[0][1], points[1][1], points[2][1], points[3][1], points[0][1]}, {{"color", "blue"}});

           
    }
};

class TRoad {
public:
    struct Boundary {
        double up, down, left, right;
    } boundary;

    double road_length;
    double x_res, y_res, v_res, t_res;
    double x_max, y_max, v_max, t_max;

    int xCell_max, yCell_max, vCell_max, tCell_max;

    TRoad(const std::vector<double>& boundary, double road_length, const std::vector<double>& resolution, const std::vector<double>& limits)
        : road_length(road_length) {
        this->boundary.up = boundary[0];
        this->boundary.down = boundary[1];
        this->boundary.left = boundary[2];
        this->boundary.right = boundary[3];

        x_res = resolution[0];
        y_res = resolution[1];
        v_res = resolution[2];
        t_res = resolution[3];

        x_max = limits[0];
        y_max = limits[1];
        v_max = limits[2];
        t_max = limits[3];

        xCell_max = static_cast<int>(x_max / x_res);
        yCell_max = static_cast<int>(y_max / y_res);
        vCell_max = static_cast<int>(v_max / v_res);
        tCell_max = static_cast<int>(t_max / t_res);
    }

    bool out_road(const State& s0) const {

        return (s0.y > boundary.up - 1.2 ||
                (s0.x < boundary.left + 1.2 && s0.y < boundary.down + 1.2) ||
                (s0.x > boundary.right -1.2 && s0.y < boundary.down + 1.2));
    }

    std::vector<double> get_goal(int goal) const {
        switch (goal) {
            case 0:
                return {-9, boundary.up - road_length / 2.0};
            case 1:
                return {boundary.left + road_length / 2.0, -9};
            case 2:
                return {9, boundary.down + road_length / 2.0};
            default:
                std::cout << "invalid goal!\n";
                return {0, 0};
        }
    }

    std::string get_index(const std::vector<State>& cars_state) const {
        std::ostringstream index;

        for (const auto& s0 : cars_state) {
            int ind = 0;

            double x = std::clamp(s0.x, -x_max / 2.0, x_max / 2.0) + x_max / 2.0;
            double y = std::clamp(s0.y, -y_max / 2.0, y_max / 2.0) + y_max / 2.0;
            double v = std::clamp(s0.v, -v_max / 2.0, v_max / 2.0) + v_max / 2.0;
            double theta = std::clamp(s0.theta, -t_max / 2.0, t_max / 2.0) + t_max / 2.0;

            int x_cell = static_cast<int>(x / x_res);
            int y_cell = static_cast<int>(y / y_res);
            int v_cell = static_cast<int>(v / v_res);
            int t_cell = static_cast<int>(theta / t_res);

            ind += x_cell + y_cell * xCell_max + v_cell * xCell_max * yCell_max + t_cell * xCell_max * yCell_max * vCell_max;

            index << ind << ' ';
        }

        std::string index_str = index.str();
        index_str.pop_back(); // Remove trailing space

        return index_str;
    }

    void draw(){
        plt::plot({-10, 10}, {boundary.up, boundary.up}, {{"color", "red"}});
        plt::plot({-10, boundary.left}, {boundary.down, boundary.down}, {{"color", "red"}});
        plt::plot({boundary.right, 10}, {boundary.down, boundary.down}, {{"color", "red"}});
        plt::plot({boundary.left,boundary.left}, {-10, boundary.down}, {{"color", "red"}});
        plt::plot({boundary.right,boundary.right}, {-10, boundary.down}, {{"color", "red"}});
    }
};

class Generator {
public:
    std::vector<Car> sim_cars;
    TRoad road_env;

    Generator(double dt, double width, double length, int num, const TRoad& road_env)
        : road_env(road_env) {
        sim_cars.reserve(num);
        for (int i = 0; i < num; ++i) {
            sim_cars.emplace_back(dt, width, length);
        }
    }

    bool gen(std::vector<State>& states, const std::string& action, std::string &obs_index, double &reward) {
        std::vector<int> acts;
        std::istringstream iss(action);
        int act;
        while (iss >> act) {
            acts.push_back(act);
        }

        std::vector<State> last_state;
        last_state = states;

        for (size_t i = 0; i < sim_cars.size(); ++i) {
            sim_cars[i].step(states[i], acts[i]);
        }

        obs_index = road_env.get_index(states);
        reward = get_reward(states, acts, last_state);

        return true;
    }

    std::vector<double> getGoal(int &flag){
        return road_env.get_goal(flag);
    }

private:
    double get_reward(const std::vector<State> &s, const std::vector<int>& acts, const std::vector<State> &last_s) const {
        double reward = 0.0;
        size_t car_num = sim_cars.size();

        for (size_t i = 0; i < car_num; ++i) {
            const State& s0 = s[i];
            const State& last_s0 = last_s[i];
            std::vector<double> goal = road_env.get_goal(s0.g);
            std::vector<double> cur_pos = {s0.x, s0.y};
            std::vector<double> last_pos = {last_s0.x, last_s0.y};

            // goal
            auto last_dist = std::sqrt(std::pow(last_pos[0] - goal[0], 2) + std::pow(last_pos[1] - goal[1], 2));
            auto cur_dist =  std::sqrt(std::pow(cur_pos[0] - goal[0], 2) + std::pow(cur_pos[1] - goal[1], 2));

            if (cur_dist > 0.6){
              if (cur_dist < last_dist) {
                reward += 40 / cur_dist;
              } else if (cur_dist >= last_dist) {
                reward -= 0.1;
              }
            } else {
              if (acts[i] != STOP) {
                reward -= 1;
              } else {
                  reward += 1;
              }
            }
          if (acts[i] != STOP) {
            reward -= 0.01;
          }

          // collision
          bool col = false;
          if (i + 1 < car_num) {
            for (size_t j = i + 1; j < car_num; ++j) {
              auto s0 = s[i];
              auto s1 = s[j];

              // AABB (Axis-Aligned Bounding Box) collision detection
              double dx = fabs(s0.x - s1.x);
              double dy = fabs(s0.y - s1.y);

              if (dx < sim_cars[i].length && dy < sim_cars[i].width) {
                // std::cout << "collision\n";
                col = true;
              }
            }
            }

            if (col) {
                reward -= 20;
            }

            // road
            if (road_env.out_road(s[i])) {
                reward -= 20;
            }
        }

        return reward;
    }
};

#endif
