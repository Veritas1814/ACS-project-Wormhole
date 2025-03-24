#include "decision_tree_op5.h"
#include <fstream>
#include <iostream>
#include <queue>
#include <algorithm>

using json = nlohmann::json;

void DecisionTreeOp5::loadFromJson(const std::string& filename) {
    std::ifstream file(filename);
    if(!file.is_open()){
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

int DecisionTreeOp5::buildTree(const json& treeData, int index) {
    NodeOp5 node;
    if (treeData["children_left"][index] == -1) {
        node.isLeaf = true;
        const auto& values = treeData["value"][index][0];
        int maxIdx = 0;
        for (int i = 1; i < values.size(); ++i) {
            if (values[i] > values[maxIdx])
                maxIdx = i;
        }
        node.value = maxIdx;
        node.feature = -1;
        node.threshold = 0.0;
    } else {
        node.isLeaf = false;
        node.feature = treeData["feature"][index];
        node.threshold = treeData["threshold"][index];
        int leftIndex = buildTree(treeData, treeData["children_left"][index]);
        int rightIndex = buildTree(treeData, treeData["children_right"][index]);
    }
    nodes.push_back(node);
    return nodes.size() - 1;
}

std::string DecisionTreeOp5::predict(const std::vector<double>& sample) {
    int cur = 0;
    while (!nodes[cur].isLeaf) {
        cur = (sample[nodes[cur].feature] >= nodes[cur].threshold) ? 2 * cur + 2 : 2 * cur + 1;
    }
    return classLabels[nodes[cur].value];
}
