#include "decision_tree_op2.h"
#include <fstream>
#include <iostream>

void DecisionTreeOp2::loadFromJson(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()){
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    json treeData;
    file >> treeData;
    auto tree = treeData["tree"];
    classLabels = tree["classes"].get<std::vector<std::string>>();
    root = buildTree(tree, 0);
}

std::shared_ptr<DecisionTreeOp2::NodeOp2> DecisionTreeOp2::buildTree(const json& treeData, int index) {
    auto node = std::make_shared<DecisionTreeOp2::NodeOp2>();
    if (treeData["children_left"][index] == -1) {
        node->isLeaf = true;
        const auto& values = treeData["value"][index][0];
        int maxIdx = 0;
        for (int i = 1; i < values.size(); ++i) {
            if (values[i] > values[maxIdx])
                maxIdx = i;
        }
        node->value = maxIdx;
    } else {
        node->isLeaf = false;
        node->feature = treeData["feature"][index];
        node->threshold = treeData["threshold"][index];
        node->left = buildTree(treeData, treeData["children_left"][index]);
        node->right = buildTree(treeData, treeData["children_right"][index]);
    }
    return node;
}

int DecisionTreeOp2::predict(const std::vector<float>& sample) noexcept {
    auto node = root;
    while (!node->isLeaf) {
        node = (sample[node->feature] < node->threshold) ? node->left : node->right;
    }
    return node->value;
}
