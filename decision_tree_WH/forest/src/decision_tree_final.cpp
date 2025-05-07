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
    features.clear();
    thresholds.clear();
    values.clear();
    isLeaf.clear();
    leftIndices.clear();
    rightIndices.clear();
    buildTree(treeData["tree"]);
}

void DecisionTreeFinal::buildTree(const json& treeData) {
    size_t nNodes = treeData["children_left"].size();

    features.resize(nNodes, -1);
    thresholds.resize(nNodes, 0.0);
    values.resize(nNodes, -1);
    isLeaf.resize(nNodes, false);
    leftIndices.resize(nNodes, -1);
    rightIndices.resize(nNodes, -1);

    for (size_t idx = 0; idx < nNodes; idx++) {
        int left = treeData["children_left"][idx];
        int right = treeData["children_right"][idx];

        if (left == -1 && right == -1) {
            // It's a leaf node
            isLeaf[idx] = true;
            const auto& vals = treeData["value"][idx][0];
            values[idx] = std::distance(vals.begin(), std::max_element(vals.begin(), vals.end()));
        } else {
            // It's an internal node; enforce both left and right exist
            if (left == -1 || right == -1) {
                std::cerr << "Warning: Node " << idx << " is missing a child. Tree is not full.\n";
                // Fill with dummy child that copies the current node’s prediction
                isLeaf[idx] = true;
                const auto& vals = treeData["value"][idx][0];
                values[idx] = std::distance(vals.begin(), std::max_element(vals.begin(), vals.end()));
                continue;
            }

            features[idx] = treeData["feature"][idx];
            thresholds[idx] = treeData["threshold"][idx];
            leftIndices[idx] = left;
            rightIndices[idx] = right;
        }
    }
}

int DecisionTreeFinal::predict(const std::vector<float>& sample) {
    int cur = 0;
    while (!isLeaf[cur]) {
        cur = (sample[features[cur]] < thresholds[cur]) ? leftIndices[cur] : rightIndices[cur];
    }
    return values[cur];
}