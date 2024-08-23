#ifndef POMCP_HPP_
#define POMCP_HPP_

#include "auxilliary.hpp"
#include "env.hpp" // Assuming this contains the State, Car, and other related classes
#include <algorithm>
#include <cmath>
#include <ctime>
#include <random>
#include <string>

class MCTS {
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
  MCTS(Generator *generator, double gamma = 0.95, double c = 1,
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

  std::pair<std::string, unsigned int> SearchBest(const unsigned int &h, const int & g, const bool UseUCB = true) {
    double max_value = -std::numeric_limits<double>::infinity();
    unsigned int result = 0;
    int max_Nc = -1;
    std::string resulta;

    if (UseUCB) {
      for (const auto &[action, child] : tree.nodes[h].children) {
        if (tree.nodes[child].Nc[g] == 0) {
          return {action, child};
        }
        double ucb =
            UCB(tree.nodes[h].Nc[g], tree.nodes[child].Nc[g], tree.nodes[child].V[g], c);
        if (ucb > max_value) {
          max_value = ucb;
          result = child;
          resulta = action;
        }
      }
    } else {

      // WARN: randomly select intention
      // auto s = Bh.empty() ? states[rand() % states.size()] : Bh[rand() % Bh.size()];
      const auto & Bh = tree.nodes[h].Bh;
      std::vector<double> pr(states.size(), 1.0 / states.size());
      if (!Bh.empty()) {
        std::vector<int> particles(states.size(), 0);
        for (int i = 0; i < Bh.size(); i++) {
          particles[Bh[i][0]]++;
        }

        for (size_t i = 0; i < states.size(); i++){
          pr[i] = static_cast<double>(particles[i]) / Bh.size();
        }
      }

      int max_g = std::max_element(pr.begin(), pr.end()) - pr.begin();
      for (const auto &[action, child] : tree.nodes[h].children) {
        // std::cout << "action: " << action << ", Nc: " <<
        // tree.nodes[child].Nc[g] << ", V: " << tree.nodes[child].V[g] <<
        // std::endl;
        double node_value = 0;
        int node_Nc = 0;

        // WARN: use contingency
        for (int i = 0; i < pr.size(); i++){
          node_value += tree.nodes[child].V[i] * pr[i];
          node_Nc += tree.nodes[child].Nc[i];
        }

        // WARN: use max
        // node_value = tree.nodes[child].V[max_g];
        // node_Nc = tree.nodes[child].Nc[max_g];

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
    auto [action, _] = SearchBest(0, -1, false);

    // std::cout << "g: " << s[0] << ", act: " << action << std::endl;

    std::istringstream iss(action);
    int act;
    iss >> act;
    return act;
  }

  int getObservationNode(const unsigned int &h, const std::string &act_obs) {
    return tree.getObservationNode(h, act_obs);
  }

  std::pair<std::string, double> Rollout(std::vector<State> &s, int depth) {
    if (pow(gamma, depth) < e && depth != 0) {
      return {"", 0};
    }

    auto action = actions[rand() % actions.size()];

    std::string obs_index;
    double reward;
    generator->gen(s, action, obs_index, reward);
    reward += gamma * Rollout(s, depth + 1).second;
    return {action, reward};
  }

  double Simulate(std::vector<State> &s, unsigned int h, int depth) {
    if (tree.nodes.find(h) == tree.nodes.end()) {
      throw std::runtime_error("Invalid node index during simulation: " +
                               std::to_string(h));
    }

    if (pow(gamma, depth) < e && depth != 0) {
      return 0;
    }

    // WARN: only one agent's intention is implemented now
    auto g = s[1].g;

    if (tree.isLeafNode(h, g)) {
      for (auto &action : actions) {
        tree.ExpandTreeFrom(h, action);
      }
      tree.nodes[h].Nc[g]++;
      tree.nodes[h].Bh.push_back({g});

      auto a_r = Rollout(s, depth);
      auto ha = getObservationNode(h, a_r.first);
      tree.nodes[ha].Nc[g]++;
      tree.nodes[ha].V[g] = a_r.second;
      return a_r.second;
    }

    auto [next_action, ha] = SearchBest(h, g);

    std::string obs_index;
    double reward = 0;

    generator->gen(s, next_action, obs_index, reward);


    auto hao = getObservationNode(ha, obs_index);
    reward += gamma * Simulate(s, hao, depth + 1);

    // std::cout << "action: " << next_action << ", reward: " << reward << std::endl;

    tree.nodes[h].Bh.push_back({g});
    if (tree.nodes[h].Bh.size() > no_particles) {
      tree.nodes[h].Bh.erase(tree.nodes[h].Bh.begin());
    }
    tree.nodes[h].Nc[g]++;
    tree.nodes[ha].Nc[g]++;
    tree.nodes[ha].V[g] += (reward - tree.nodes[ha].V[g]) / tree.nodes[ha].Nc[g];

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

  std::vector<double> UpdateBelief(const std::string &action, const std::string &observation) {
    std::cout << "update belief\n";
    auto prior = tree.nodes[0].Bh;

    // for (auto & [act, nex] : tree.nodes[0].children){
    //   std::cout << "act: " << act << ", Nc: " <<  tree.nodes[nex].Nc << "\n";
    // }

    auto action_node = tree.nodes[0].children[action];
    auto next_node = getObservationNode(action_node, observation);
    // while (tree.nodes[next_node].Bh.size() < no_particles / 2 ) {
    //   // tree.nodes[next_node].Bh.push_back(
    //   //     PosteriorSample(prior, action, observation));
    // }
    for (int i = 0; i < 5; i++){
      tree.nodes[next_node].Bh.push_back({0});
      tree.nodes[next_node].Bh.push_back({1});
      tree.nodes[next_node].Bh.push_back({2});
    }

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

    std::vector<double> res;
    double total = a0 + a1 + a2;
    res.push_back(a0 / total);
    res.push_back(a1 / total);
    res.push_back(a2 / total);

    return res;
  }
};

#endif
