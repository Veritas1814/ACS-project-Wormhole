#include "decision_tree_op4.h"
#include <fstream>
#include <iostream>
#include <queue>

void DecisionTreeOp4::loadFromJson(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    json treeData;
    file >> treeData;
    classLabels = treeData["tree"]["classes"].get<std::vector<std::string>>();
    nodes.clear();
    buildTree(treeData["tree"]);
}

void DecisionTreeOp4::buildTree(const json& treeData) {
    std::queue<int> q;
    q.push(0);
    while (!q.empty()) {
        int idx = q.front();
        q.pop();
        NodeOp4 node;
        if (treeData["children_left"][idx] == -1) {
            node.isLeaf = true;
            const auto& vals = treeData["value"][idx][0];
            int maxIdx = std::max_element(vals.begin(), vals.end()) - vals.begin();
            node.value = maxIdx;
            node.feature = -1;
            node.threshold = 0.0;
            node.leftIndex = -1;
            node.rightIndex = -1;
        } else {
            node.isLeaf = false;
            node.feature = treeData["feature"][idx];
            node.threshold = treeData["threshold"][idx];
            q.push(treeData["children_left"][idx]);
            q.push(treeData["children_right"][idx]);
            node.leftIndex = treeData["children_left"][idx];
            node.rightIndex = treeData["children_right"][idx];
        }
        nodes.push_back(node);
    }
}

std::string DecisionTreeOp4::predict(const std::vector<double>& sample) {
    int cur = 0;
    while (!nodes[cur].isLeaf) {
        cur = (sample[nodes[cur].feature] < nodes[cur].threshold)
                  ? nodes[cur].leftIndex
                  : nodes[cur].rightIndex;
    }
    return classLabels[nodes[cur].value];
}
