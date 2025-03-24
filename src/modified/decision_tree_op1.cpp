// decision_tree_op1.cpp
#include "../include/decision_tree.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct NodeOp1 {
    int feature;
    double threshold;
    int value;
    bool isLeaf;
    int leftIndex;
    int rightIndex;
};

class DecisionTreeOp1 {
public:
    std::vector<NodeOp1> nodes;
    std::vector<std::string> classLabels;

    void loadFromJson(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()){
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }
        json treeData;
        file >> treeData;
        auto tree = treeData["tree"];
        classLabels = tree["classes"].get<std::vector<std::string>>();
        nodes.clear();
        buildTree(tree, 0);
    }

    // Recursively build the tree and return the index of the created node.
    int buildTree(const json& treeData, int index) {
        NodeOp1 node;
        if (treeData["children_left"][index] == -1) {
            node.isLeaf = true;
            const auto& values = treeData["value"][index][0];
            int maxIdx = 0;
            for (int i = 1; i < values.size(); ++i) {
                if (values[i] > values[maxIdx])
                    maxIdx = i;
            }
            node.value = maxIdx;
        } else {
            node.isLeaf = false;
            node.feature = treeData["feature"][index];
            node.threshold = treeData["threshold"][index];
            node.leftIndex = buildTree(treeData, treeData["children_left"][index]);
            node.rightIndex = buildTree(treeData, treeData["children_right"][index]);
        }
        nodes.push_back(node);
        return nodes.size() - 1;
    }

    std::string predict(const std::vector<double>& sample) {
        int cur = 0;
        while (!nodes[cur].isLeaf) {
            cur = (sample[nodes[cur].feature] < nodes[cur].threshold)
                      ? nodes[cur].leftIndex
                      : nodes[cur].rightIndex;
        }
        return classLabels[nodes[cur].value];
    }
};
