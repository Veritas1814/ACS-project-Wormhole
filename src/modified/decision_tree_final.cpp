#include "decision_tree_final.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <future>

using json = nlohmann::json;

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
    size_t numNodes = treeData["feature"].size();
    nodes.resize(numNodes);

    auto buildSubtree = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            if (treeData["children_left"][i] == -1) { // Leaf node
                nodes[i].isLeaf = true;
                const auto& vals = treeData["value"][i][0];
                nodes[i].value = std::distance(vals.begin(), std::max_element(vals.begin(), vals.end()));
                nodes[i].feature = -1;
                nodes[i].threshold = 0.0;
                nodes[i].leftIndex = -1;
                nodes[i].rightIndex = -1;
            } else {
                nodes[i].isLeaf = false;
                nodes[i].feature = treeData["feature"][i];
                nodes[i].threshold = treeData["threshold"][i];
                nodes[i].leftIndex = treeData["children_left"][i];
                nodes[i].rightIndex = treeData["children_right"][i];
                nodes[i].value = -1;
            }
        }
    };

    size_t mid = numNodes / 2;
    auto leftFuture = std::async(std::launch::async, buildSubtree, 0, mid);
    auto rightFuture = std::async(std::launch::async, buildSubtree, mid, numNodes);

    leftFuture.get();
    rightFuture.get();
}

std::string DecisionTreeFinal::predict(const std::vector<double>& sample) {
    int nodeIndex = 0;
    while (!nodes[nodeIndex].isLeaf) {
        nodeIndex = (sample[nodes[nodeIndex].feature] < nodes[nodeIndex].threshold)
                    ? nodes[nodeIndex].leftIndex
                    : nodes[nodeIndex].rightIndex;
    }
    return classLabels[nodes[nodeIndex].value];
}
