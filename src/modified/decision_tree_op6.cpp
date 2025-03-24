#include "decision_tree_op6.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <future>

using json = nlohmann::json;

NodeOp6::NodeOp6() : feature(-1), threshold(0.0), value(-1), isLeaf(false), left(nullptr), right(nullptr) {}

void DecisionTreeOp6::loadFromJson(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    json treeData;
    file >> treeData;
    auto tree = treeData["tree"];
    classLabels = tree["classes"].get<std::vector<std::string>>();

    root = buildTree(tree, 0);
}

std::shared_ptr<NodeOp6> DecisionTreeOp6::buildTree(const json& treeData, int index) {
    auto node = std::make_shared<NodeOp6>();
    if (treeData["children_left"][index] == -1) {
        node->isLeaf = true;
        const auto& vals = treeData["value"][index][0];
        int maxIdx = 0;
        for (int i = 1; i < vals.size(); ++i) {
            if (vals[i] > vals[maxIdx])
                maxIdx = i;
        }
        node->value = maxIdx;
    } else {
        node->isLeaf = false;
        node->feature = treeData["feature"][index];
        node->threshold = treeData["threshold"][index];

        // Launch asynchronous tasks to build left and right subtrees concurrently
        auto leftFuture = std::async(std::launch::async, &DecisionTreeOp6::buildTree, this, std::cref(treeData), treeData["children_left"][index]);
        auto rightFuture = std::async(std::launch::async, &DecisionTreeOp6::buildTree, this, std::cref(treeData), treeData["children_right"][index]);

        // Wait for the tasks to complete and get the results
        node->left = leftFuture.get();
        node->right = rightFuture.get();
    }
    return node;
}

std::string DecisionTreeOp6::predict(const std::vector<double>& sample) {
    auto node = root;
    while (!node->isLeaf) {
        node = (sample[node->feature] < node->threshold) ? node->left : node->right;
    }
    return classLabels[node->value];
}
