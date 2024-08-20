#ifndef POMCP_HPP_
#define POMCP_HPP_

#include "auxilliary.hpp"
#include "env.hpp" // Assuming this contains the State, Car, and other related classes
#include <cmath>
#include <ctime>
#include <random>
#include <string>

class POMCP {
private:
  double gamma;
  double e;
  double c;
  int timeout;
  int no_particles;
  BuildTree tree;
  std::vector<std::vector<int>> states;
  std::vector<std::string> actions;
  std::vector<std::string> observations;
  Generator *generator;

public:
  POMCP(Generator *generator, double gamma = 0.95, double c = 1,
        double threshold = 0.005, int timeout = 10000, int no_particles = 1200)
      : gamma(gamma), c(c), e(threshold), timeout(timeout),
        no_particles(no_particles), generator(generator), tree(BuildTree()) {
    if (gamma >= 1) {
      throw std::invalid_argument("gamma should be less than 1.");
    }
  }

  void initialize(const std::vector<std::vector<int>> &S,
                  const std::vector<std::string> &A,
                  const std::vector<std::string> &O) {
    states = S;
    actions = A;
    observations = O;
  }

  std::pair<std::string, unsigned int> SearchBest(const unsigned int &h, bool UseUCB = true) {
    double max_value = -std::numeric_limits<double>::infinity();
    unsigned int result = 0;
    int max_Nc = -1;
    std::string resulta = "0 0 0";

    if (UseUCB) {
      for (const auto &[action, child] : tree.nodes[h].children) {
        // if (tree.nodes[child].Nc == 0) {
        //   return {action, child};
        // }
        double ucb =
            UCB(tree.nodes[h].Nc, tree.nodes[child].Nc + 1, tree.nodes[child].V, c);
        if (ucb > max_value) {
          max_value = ucb;
          result = child;
          resulta = action;
        }
      }
    } else {
      for (const auto &[action, child] : tree.nodes[h].children) {
        // std::cout << "action: " << action << ", Nc: " << tree.nodes[child].Nc << ", V: " << tree.nodes[child].V << std::endl;
        double node_value = tree.nodes[child].V;
        int node_Nc = tree.nodes[child].Nc;
        if (max_value < node_value) {
          max_value = node_value;
          result = child;
          resulta = action;
          max_Nc = node_Nc;
          // std::cout << "value: " << max_value << std::endl;
        }
      }
    }
    return {resulta, result};
  }

  int Search(const std::vector<State> &cur_state) {
    auto Bh = tree.nodes[0].Bh;
    std::vector<State> tmp_state;
    for (int j = 0; j < timeout; ++j) {
      auto s =
          Bh.empty() ? states[rand() % states.size()] : Bh[rand() % Bh.size()];

      tmp_state = cur_state;
      for (int i = 0; i < int(s.size()); ++i) {
        tmp_state[i + 1].g = s[i];
      }

      Simulate(tmp_state, 0, 0);

      // for (const auto &[action, child] : tree.nodes[0].children) {
      //   double node_value = tree.nodes[child].V;
      //   std::cout << "action2: " << action << "value: " << node_value << std::endl;
      // }
    }
    auto [action, _] = SearchBest(0, false);

    std::cout << "act: " << action << std::endl;

    std::istringstream iss(action);
    int act;
    iss >> act;
    // while (iss >> act) {
    //     acts.push_back(act);
    // }
    return act;
  }

  int getObservationNode(const unsigned int &h, const std::string &sample_observation) {
    return tree.getObservationNode(h, sample_observation);
  }

  double Rollout(std::vector<State> &s, int depth) {
    
    if (pow(gamma, depth) < e && depth != 0) {
      return 0;
    }

    double cum_reward = 0;
    auto action = actions[rand() % actions.size()];
    // std::string action = "1 1";

    // std::vector<int> actions;
    // for (int i = 0; i < s.size(); i++){
    //   auto goal = generator->getGoal(s[i].g);

    //   auto goalPose = std::atan2(goal[1] - s[i].y, goal[2] - s[i].x);

    //   if (cos(goalPose - s[i].theta) >0.9 && std::sqrt(std::pow(goal[0] - s[i].x, 2) + std::pow(goal[1] - s[i].y, 2)) > 1){
    //       actions.push_back(1);
    //   } else if (std::sqrt(std::pow(goal[0] - s[i].x, 2) + std::pow(goal[1] - s[i].y, 2)) <= 1){
    //     actions.push_back(0);
    //   } else if (cos(goalPose - s[i].theta) <= 0.9 ){
    //     auto next_theta1 = goalPose - (s[i].theta + M_PI / 8);
    //     auto next_theta2 = goalPose - (s[i].theta - M_PI / 8);
    //     if (cos(next_theta1) > cos(next_theta2)){
    //       actions.push_back(4);
    //     } else {
    //       actions.push_back(3);
    //     }
    //   } else {
    //     actions.push_back(0);
    //   }
    // }

    // std::string action = std::to_string(actions[0]) + " " + std::to_string(actions[1]);

    std::string obs_index;
    double reward;
    generator->gen(s, action, obs_index, reward);
    cum_reward += reward + gamma * Rollout(s, depth + 1);

    // std::cout << "rollout: " << depth << std::endl;
    return cum_reward;
  }

  double Simulate(std::vector<State> &s, unsigned int h, int depth) {

    if (tree.nodes.find(h) == tree.nodes.end()) {
      throw std::runtime_error("Invalid node index during simulation: " +
                               std::to_string(h));
    }

    if (pow(gamma, depth) < e && depth != 0) {
      return 0;
    }

    if (tree.isLeafNode(h)) {
      for (auto &action : actions) {
        tree.ExpandTreeFrom(h, action);
      }
      double new_value = Rollout(s, depth);
      tree.nodes[h].Nc++;
      tree.nodes[h].Bh.push_back({s[1].g});
      // std::cout << "h: " << h << "value: " << new_value << std::endl;
      
      return new_value;
    }

    auto [next_action, ha] = SearchBest(h);

    std::string obs_index;
    double reward = 0;

    generator->gen(s, next_action, obs_index, reward);


    auto hao = getObservationNode(ha, obs_index);
    reward += gamma * Simulate(s, hao, depth + 1);

    // std::cout << "action: " << next_action << ", reward: " << reward << std::endl;

    // tree.nodes[h].Bh.push_back({s[1].g, s[2].g});
    tree.nodes[h].Bh.push_back({s[1].g});
    // if (tree.nodes[h].Bh.size() > no_particles) {
    //   tree.nodes[h].Bh.erase(tree.nodes[h].Bh.begin());
    // }
    tree.nodes[h].Nc++;
    tree.nodes[ha].Nc++;
    tree.nodes[ha].V += (reward - tree.nodes[ha].V) / tree.nodes[ha].Nc;
    // std::cout << "dp: " << depth << ", ha: " << ha << ", V: " << tree.nodes[ha].V << std::endl;
   
    return reward;
  }

  std::vector<int> PosteriorSample(const std::vector<std::vector<int>> &Bh,
                                   const std::string &action,
                                   const std::string &observation) {
    if (Bh.empty()) {
      return states[rand() % states.size()];
    }
    return Bh[rand() % Bh.size()];
  }

  void UpdateBelief(const std::string &action, const std::string &observation) {
    std::cout << "update belief\n";
    auto prior = tree.nodes[0].Bh;

    // for (auto & [act, nex] : tree.nodes[0].children){
    //   std::cout << "act: " << act << ", Nc: " <<  tree.nodes[nex].Nc << "\n";
    // }

    auto action_node = tree.nodes[0].children[action];

    auto next_node = getObservationNode(action_node, observation);

    // while (tree.nodes[next_node].Bh.size() < no_particles ) {
    //   tree.nodes[next_node].Bh.push_back(
    //       PosteriorSample(prior, action, observation));
    // }

    tree.prune_after_action(action, observation);
    int a0, a1, a2;
    a0 = a1 = a2 = 0;
    for(int i = 0; i < tree.nodes[0].Bh.size(); i++){
      switch(tree.nodes[0].Bh[i][0]){
        case 0:{
            a0++;
            break;
          }
        case 1: {
            a1++;
            break;
          }
        case 2: {
            a2++;
            break;
          }
        default:
          break;
      }
    }
    std::cout << a0 << " : " << a1 << " : " << a2 << std::endl;
  }
};

#endif
