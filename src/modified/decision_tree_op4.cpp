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
    size_t nNodes = treeData["children_left"].size();
    nodes.resize(nNodes);

    // Initialize nodes with shared pointers
    for (size_t idx = 0; idx < nNodes; idx++) {
        nodes[idx] = std::make_shared<NodeOp4>();
    }

    for (size_t idx = 0; idx < nNodes; idx++) {
        auto& node = nodes[idx];

        if (treeData["children_left"][idx] == -1) {
            node->isLeaf = true;
            const auto& vals = treeData["value"][idx][0];
            node->value = std::distance(vals.begin(), std::max_element(vals.begin(), vals.end()));
            node->feature = -1;
            node->threshold = 0.0;
            node->leftIndex = nullptr;
            node->rightIndex = nullptr;
        } else {
            node->isLeaf = false;
            node->feature = treeData["feature"][idx];
            node->threshold = treeData["threshold"][idx];
            node->leftIndex = nodes[treeData["children_left"][idx]];
            node->rightIndex = nodes[treeData["children_right"][idx]];
        }
    }
}


std::string DecisionTreeOp4::predict(const std::vector<double>& sample) {
    std::shared_ptr<NodeOp4> cur = nodes[0];
    while (!cur->isLeaf) {
        cur = (sample[cur->feature] < cur->threshold) ? cur->leftIndex : cur->rightIndex;
    }
    return classLabels[cur->value];
}