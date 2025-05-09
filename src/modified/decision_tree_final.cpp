#include "decision_tree_final.h"
#include <fstream>
#include <iostream>
#include <queue>

void DecisionTreeFinal::loadFromJson(const std::string& filename) {
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

void DecisionTreeFinal::buildTree(const json& treeData) {
    size_t nNodes = treeData["children_left"].size();
    nodes.resize(nNodes);
    for (size_t idx = 0; idx < nNodes; idx++) {
        NodeFinal node;
        if (treeData["children_left"][idx] == -1) {
            node.isLeaf = true;
            const auto& vals = treeData["value"][idx][0];
            node.value = std::distance(vals.begin(), std::max_element(vals.begin(), vals.end()));
            node.feature = -1;
            node.threshold = 0.0;
            node.leftIndex = -1;
            node.rightIndex = -1;
        } else {
            node.isLeaf = false;
            node.feature = treeData["feature"][idx];
            node.threshold = treeData["threshold"][idx];
            node.leftIndex = treeData["children_left"][idx];
            node.rightIndex = treeData["children_right"][idx];
        }
        nodes[idx] = node;
    }
}


int DecisionTreeFinal::predict(const std::vector<float>& sample) noexcept {
    int cur = 0;
    while (!nodes[cur].isLeaf) {
        cur = (sample[nodes[cur].feature] < nodes[cur].threshold)
                  ? nodes[cur].leftIndex
                  : nodes[cur].rightIndex;
    }
    return nodes[cur].value;
}