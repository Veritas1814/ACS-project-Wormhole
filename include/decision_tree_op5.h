#pragma once

#include <vector>
#include <string>
#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;


class DecisionTreeOp5 {
public:

    void loadFromJson(const std::string& filename);
    void buildTree(const json& treeData, int index);
    std::string predict(const std::vector<double>& sample) const;

private:
    struct NodeOp5 {
        int feature;
        double threshold;
        int value;
        bool isLeaf;
        int leftIndex;
        int rightIndex;

    };

    std::vector<NodeOp5> nodes;
    std::vector<std::string> classLabels;
};
