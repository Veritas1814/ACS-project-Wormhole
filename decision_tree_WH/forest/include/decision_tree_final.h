#pragma once

#include <vector>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class DecisionTreeFinal {
public:
  void loadFromJson(const std::string& filename);
  int predict(const std::vector<float>& sample) const;
  std::vector<float> getFlatVector() const;

private:
  std::vector<int> features;
  std::vector<float> thresholds;
  std::vector<int> values;
  std::vector<std::string> classLabels;
  int depth = 0;

  void buildFlatTree(const json& treeData);
  void buildFlatRecursive(const json& treeData, int treeIdx, int flatIdx, int maxDepth, int currentDepth);
  int computeDepth(const json& treeData, int nodeIdx) const;
  void fillDummyLeaf(int idx, int predictedClass);

};
