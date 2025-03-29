#pragma once

#include <vector>
#include <string>
#include <memory>
#include <nlohmann/json.hpp>
#include <future>

using json = nlohmann::json;

class NodeOp6 {
public:
    int feature;
    double threshold;
    int value;
    bool isLeaf;
    std::shared_ptr<NodeOp6> left;
    std::shared_ptr<NodeOp6> right;

    NodeOp6();
};

class DecisionTreeOp6 {
public:
    std::shared_ptr<NodeOp6> root;
    std::vector<std::string> classLabels;

    void loadFromJson(const std::string& filename);
    std::shared_ptr<NodeOp6> buildTree(const json& treeData, int index);
    std::string predict(const std::vector<double>& sample);
};
