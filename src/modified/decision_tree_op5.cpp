// decision_tree_op5.cpp
#include "../include/decision_tree.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct NodeOp5 {
    int feature;
    double threshold;
    int value;
    bool isLeaf;
};

class DecisionTreeOp5 {
public:
    std::vector<NodeOp5> nodes;
    std::vector<std::string> classLabels;
    
    // Assume the JSON arrays are provided in level order for a complete binary tree.
    void loadFromJson(const std::string& filename) {
        std::ifstream file(filename);
        if(!file.is_open()){
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }
        json treeData;
        file >> treeData;
        auto tree = treeData["tree"];
        classLabels = tree["classes"].get<std::vector<std::string>>();
        size_t n = tree["feature"].size();
        nodes.resize(n);
        for (size_t i = 0; i < n; i++) {
            NodeOp5 node;
            if (tree["children_left"][i] == -1) {
                node.isLeaf = true;
                const auto& vals = tree["value"][i][0];
                int maxIdx = 0;
                for (int j = 1; j < vals.size(); ++j) {
                    if (vals[j] > vals[maxIdx])
                        maxIdx = j;
                }
                node.value = maxIdx;
            } else {
                node.isLeaf = false;
                node.feature = tree["feature"][i];
                node.threshold = tree["threshold"][i];
            }
            nodes[i] = node;
        }
    }
    
    std::string predict(const std::vector<double>& sample) {
        int cur = 0;
        // Use arithmetic to choose the next index:
        // left child = 2*cur + 1, right child = 2*cur + 2.
        while (!nodes[cur].isLeaf) {
            int decision = (sample[nodes[cur].feature] >= nodes[cur].threshold) ? 1 : 0;
            cur = 2 * cur + 1 + decision;
        }
        return classLabels[nodes[cur].value];
    }
};