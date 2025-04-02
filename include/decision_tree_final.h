#pragma once

#include <vector>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;


class DecisionTreeFinal {
private:
  struct NodeFinal {
    int feature;
    float threshold;
    int value;
    bool isLeaf;
    int leftIndex;
    int rightIndex;
  };

  std::vector<NodeFinal> nodes;
  std::vector<std::string> classLabels;

public:
  void loadFromJson(const std::string& filename);
  void buildTree(const json& treeData);
  std::string predict(const std::vector<double>& sample);
};