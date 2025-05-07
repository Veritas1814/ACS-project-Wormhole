#pragma once

#include <vector>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class DecisionTreeFinal {
public:
  void loadFromJson(const std::string& filename);
  int predict(const std::vector<float>& sample);

private:
  std::vector<int> features;
  std::vector<float> thresholds;
  std::vector<int> values;
  std::vector<bool> isLeaf;
  std::vector<int> leftIndices;
  std::vector<int> rightIndices;
  std::vector<std::string> classLabels;

  void buildTree(const json& treeData);
};
