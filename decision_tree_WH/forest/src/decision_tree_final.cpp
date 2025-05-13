#include "decision_tree_final.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

void DecisionTreeFinal::loadFromJson(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    json treeData;
    file >> treeData;

    classLabels = treeData["tree"]["classes"].get<std::vector<std::string>>();
    buildFlatTree(treeData["tree"]);
}

void DecisionTreeFinal::buildFlatTree(const json& treeData) {
    std::cout << "pryvit buildFlat 1" << std::endl;

    depth = computeDepth(treeData, 0, 1);
    
    size_t totalNodes = (size_t(1) << depth) - 1;
    std::cout << "pryvit bF 2 " << depth << std::endl;

    features.resize(totalNodes, -1);
    std::cout << "pryvit 555" << std::endl;

    thresholds.resize(totalNodes, 0.0f);
    std::cout << "pryvit 556" << std::endl;

    values.resize(totalNodes, -1);
    std::cout << "pryvit bf 3" << std::endl;

    buildFlatRecursive(treeData, 0, 0, depth, 1);
}

void DecisionTreeFinal::buildFlatRecursive(const json& treeData, int treeIdx, int flatIdx, int maxDepth, int currentDepth) {
    if (flatIdx >= features.size()) {
        std::cerr << "Error: flatIdx " << flatIdx << " out of range!" << std::endl;
        return;
    }

    int left = treeData["children_left"][treeIdx];
    int right = treeData["children_right"][treeIdx];

    const auto& vals = treeData["value"][treeIdx][0];
    int predictedClass = std::distance(vals.begin(), std::max_element(vals.begin(), vals.end()));

    if (left == -1 && right == -1) {
        values[flatIdx] = predictedClass;
        return;
    }

    features[flatIdx] = treeData["feature"][treeIdx];
    thresholds[flatIdx] = treeData["threshold"][treeIdx];

    int leftIdx = 2 * flatIdx + 1;
    int rightIdx = 2 * flatIdx + 2;

    if (leftIdx < features.size()) {
        if (left != -1)
            buildFlatRecursive(treeData, left, leftIdx, maxDepth, currentDepth + 1);
        else
            fillDummyLeaf(leftIdx, predictedClass);
    }

    if (rightIdx < features.size()) {
        if (right != -1)
            buildFlatRecursive(treeData, right, rightIdx, maxDepth, currentDepth + 1);
        else
            fillDummyLeaf(rightIdx, predictedClass);
    }
}

void DecisionTreeFinal::fillDummyLeaf(int idx, int predictedClass) {
    if (idx >= features.size()) return;
    features[idx] = -1;
    thresholds[idx] = 0.0f;
    values[idx] = predictedClass;
}

int DecisionTreeFinal::computeDepth(const json& treeData, int nodeIdx, int currentDepth = 1) const {
    if (currentDepth > 11)  // limit depth here
        return currentDepth;

    int left = treeData["children_left"][nodeIdx];
    int right = treeData["children_right"][nodeIdx];

    if (left == -1 && right == -1)
        return currentDepth;

    int leftDepth = (left != -1) ? computeDepth(treeData, left, currentDepth + 1) : currentDepth;
    int rightDepth = (right != -1) ? computeDepth(treeData, right, currentDepth + 1) : currentDepth;
    return std::max(leftDepth, rightDepth);
}

int DecisionTreeFinal::predict(const std::vector<float>& sample) const {
    size_t cur = 0;
    for (int level = 0; level < depth - 1; ++level) {
        int f = features[cur];
        if (f < 0 || static_cast<size_t>(f) >= sample.size()) {
            std::cerr << "Invalid feature index: " << f << " for sample size " << sample.size() << std::endl;
            return -1;
        }
        cur = 2 * cur + 1 + static_cast<size_t>(sample[f] >= thresholds[cur]);
    }

    if (cur >= values.size()) {
        std::cerr << "Error: Exceeded tree bounds during prediction. Returning -1." << std::endl;
        return -1;
    }

    return values[cur];
}


std::vector<float> DecisionTreeFinal::getFlatVector() const {
    std::vector<float> flat;

    for (size_t i = 0; i < features.size(); ++i) {
        flat.push_back(static_cast<float>(features[i]));
        flat.push_back(thresholds[i]);
        flat.push_back(static_cast<float>(values[i]));
    }

    return flat;
}
std::vector<float> DecisionTreeFinal::getFeatures() const {
    std::vector<float> float_features;
    for (size_t i = 0; i < features.size(); ++i) {
        float_features.push_back(static_cast<float>(features[i]));
    }
    return float_features;
}
std::vector<float> DecisionTreeFinal::getValues() const {
    std::vector<float> float_values;
    for (size_t i = 0; i < features.size(); ++i) {
        float_values.push_back(static_cast<float>(values[i]));
    }
    return float_values;
}
std::vector<float> DecisionTreeFinal::getTreshold() const {
    std::vector<float> float_treshold;
    for (size_t i = 0; i < features.size(); ++i) {
        float_treshold.push_back(thresholds[i]);
    }
    return float_treshold;
}