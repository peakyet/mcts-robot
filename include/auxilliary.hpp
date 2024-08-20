#ifndef AUXILLIARY_HPP_
#define AUXILLIARY_HPP_

#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <string>
#include <algorithm>
#include <iterator>

// Node structure as specified
struct Node {
    int parent;
    std::unordered_map<std::string, unsigned int> children;
    int Nc;
    double V;
    std::vector<std::vector<int>> Bh;
};

class BuildTree {
public:
    BuildTree() {
        count = 0;
        nodes.clear();

        Node initialNode;
        initialNode.parent = count;
        initialNode.Nc = 0;
        initialNode.V = -10000;
        initialNode.children.clear();
        initialNode.Bh.clear();

        // Initialize the tree with the root node
        nodes.insert({count, initialNode});
    }

    void ExpandTreeFrom(const unsigned int& parent, const std::string& index) {
        count++;
        Node newNode;
        newNode.parent = parent;
        newNode.Nc = 0;
        newNode.V = 0;
        newNode.children.clear();
        newNode.Bh.clear();
     
        nodes.insert({count, newNode});
        nodes[parent].children[index] = count;

    }

    bool isLeafNode(const unsigned int & n) const {
        const Node& node = nodes.at(n);
        return node.Nc == 0;
    }

    int getObservationNode(const unsigned int & h, const std::string& sample_observation) {

        if (nodes.find(h) == nodes.end()) {
                throw std::runtime_error("Invalid node index: " + std::to_string(h));
            }

            std::unordered_map<std::string, unsigned int> & children = nodes[h].children;

            if (children.count(sample_observation) == 0) {
                ExpandTreeFrom(h, sample_observation);
            }
            return children[sample_observation];
    }

    void prune(const unsigned int & node) {
        const auto& children = nodes[node].children;
        nodes.erase(node);
        for (const auto& [_, child] : children) {
            prune(child);
        }
    }

    void make_new_root(const unsigned int & new_root) {
        nodes[0] = nodes[new_root];
        nodes.erase(new_root);
        nodes[0].parent = 0;
        for (const auto& [_, child] : nodes[0].children) {
            nodes[child].parent = 0;
        }
    }

    void prune_after_action(const std::string& action, const std::string& observation) {
        auto action_node = nodes[0].children[action];
        auto new_root = getObservationNode(action_node, observation);

        nodes[action_node].children.erase(observation);

        prune(0);

        make_new_root(new_root);
    }

    std::unordered_map<unsigned int, Node> nodes;
private:
    unsigned int count = 0;
};

// UCB score calculation
inline double UCB(int N, int n, double V, double c = 1.0) {
    return V + c * std::sqrt(std::log(N) / n);
}

#endif
