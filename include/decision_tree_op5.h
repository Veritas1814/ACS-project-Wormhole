#pragma once

#include <vector>
#include <string>
#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct NodeOp5 {
    int feature;
    double threshold;
    int value;
    bool isLeaf;
    int leftIndex;
    int rightIndex;
};

class DecisionTreeOp5 {
public:
    std::vector<NodeOp5> nodes;
    std::vector<std::string> classLabels;

    void loadFromJson(const std::string& filename);
    void buildTree(const json& treeData, int index);
    std::string predict(const std::vector<double>& sample) const;
};