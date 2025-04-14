#include "decision_tree_op1.h"
#include <fstream>
#include <iostream>

void DecisionTreeOp1::loadFromJson(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()){
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    json treeData;
    file >> treeData;
    auto tree = treeData["tree"];
    classLabels = tree["classes"].get<std::vector<std::string>>();

    int nodeCount = tree["children_left"].size();
    nodes.resize(nodeCount); // Preallocate space

    buildTree(tree, 0);
}

int DecisionTreeOp1::buildTree(const json& treeData, int index) {
    if (index >= nodes.size()) return -1; // Avoid invalid access

    NodeOp1& node = nodes[index]; // Use reference to avoid reallocation issues

    if (treeData["children_left"][index] == -1) {
        node.isLeaf = true;
        const auto& values = treeData["value"][index][0];
        node.value = std::distance(values.begin(), std::max_element(values.begin(), values.end()));
    } else {
        node.isLeaf = false;
        node.feature = treeData["feature"][index];
        node.threshold = treeData["threshold"][index];
        node.leftIndex = buildTree(treeData, treeData["children_left"][index]);
        node.rightIndex = buildTree(treeData, treeData["children_right"][index]);
    }
    return index;
}

int DecisionTreeOp1::predict(const std::vector<double>& sample) noexcept {
    int cur = 0;
    while (!nodes[cur].isLeaf) {
        if (sample.size() <= nodes[cur].feature) {
            std::cerr << "Error: Sample data is too small for the current tree node feature index!" << std::endl;
            return -1;
        }
        cur = (sample[nodes[cur].feature] < nodes[cur].threshold)
                  ? nodes[cur].leftIndex
                  : nodes[cur].rightIndex;
    }
    return nodes[cur].value;
}
