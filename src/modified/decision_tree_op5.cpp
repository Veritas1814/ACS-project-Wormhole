// decision_tree_op5.cpp
#include "../include/decision_tree.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <nlohmann/json.hpp>

void DecisionTreeOp5::loadFromJson(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
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

void DecisionTreeOp5::buildTree(const json& treeData, int index) {
    if (index >= nodes.size()) nodes.resize(index + 1);
    NodeOp5& node = nodes[index];

    if (treeData["children_left"][index] == -1) {
        node.isLeaf = true;
        const auto& values = treeData["value"][index][0];
        node.value = std::distance(values.begin(), std::max_element(values.begin(), values.end()));
        node.feature = -1;
        node.threshold = 0.0;
    } else {
        node.isLeaf = false;
        node.feature = treeData["feature"][index];
        node.threshold = treeData["threshold"][index];
        node.leftIndex = 2 * index + 1;
        node.rightIndex = 2 * index + 2;
        buildTree(treeData, node.leftIndex);
        buildTree(treeData, node.rightIndex);
    }
}

std::string DecisionTreeOp5::predict(const std::vector<double>& sample) const {
    int cur = 0;
    while (!nodes[cur].isLeaf) {
        cur = 2 * cur + 1 + (sample[nodes[cur].feature] >= nodes[cur].threshold);
    }
};