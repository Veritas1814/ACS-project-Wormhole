#pragma once

#include "node.h"
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <memory>

using json = nlohmann::json;

class DecisionTree {
public:
    std::shared_ptr<Node> root;
    std::vector<std::string> classLabels;

    void loadFromJson(const std::string& filename);
    void loadTree(const json& treeData);
    std::shared_ptr<Node> buildTree(const json& treeData, int index);
    std::string predict(const std::vector<double>& sample);
};
