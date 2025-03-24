#include "../include/decision_tree.h"
#include <fstream>
#include <iostream>
#include <algorithm> // for std::max_element

void DecisionTree::loadFromJson(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    json treeData;
    file >> treeData;

    if (treeData.find("tree") == treeData.end()) {
        std::cerr << "Error: No 'tree' data found in JSON file." << std::endl;
        return;
    }

    auto tree = treeData["tree"];

    if (tree.find("classes") == tree.end()) {
        std::cerr << "Error: No 'classes' data found in tree." << std::endl;
        return;
    }

    classLabels = tree["classes"].get<std::vector<std::string>>();
    loadTree(tree);
}

std::shared_ptr<Node> DecisionTree::buildTree(const json& treeData, int index) {
    auto node = std::make_shared<Node>();

    // Check if current node is a leaf node
    if (treeData["children_left"][index] == -1) {
        node->isLeaf = true;
        const auto& values = treeData["value"][index][0];

        // Find the class with the highest count
        node->value = std::distance(values.begin(), std::max_element(values.begin(), values.end()));

        if (values.empty()) {
            std::cerr << "Error: Value array is empty for leaf node at index " << index << std::endl;
        }
    } else {
        node->isLeaf = false;
        node->feature = treeData["feature"][index];
        node->threshold = treeData["threshold"][index];

        // Ensure children exist
        if (treeData["children_left"].size() <= index || treeData["children_right"].size() <= index) {
            std::cerr << "Error: Invalid children data for node at index " << index << std::endl;
            return nullptr;
        }

        node->left = buildTree(treeData, treeData["children_left"][index]);
        node->right = buildTree(treeData, treeData["children_right"][index]);
    }

    return node;
}

void DecisionTree::loadTree(const json& treeData) {
    if (treeData.find("children_left") == treeData.end() ||
        treeData.find("children_right") == treeData.end()) {
        std::cerr << "Error: Missing necessary tree structure fields in JSON." << std::endl;
        return;
    }

    root = buildTree(treeData, 0);
}

std::string DecisionTree::predict(const std::vector<double>& sample) {
    auto node = root;

    while (node && !node->isLeaf) {
        if (sample.size() <= node->feature) {
            std::cerr << "Error: Sample data is too small for the current tree node feature index!" << std::endl;
            return "";
        }

        node = (sample[node->feature] < node->threshold) ? node->left : node->right;
    }

    return node ? classLabels[node->value] : "";
}
