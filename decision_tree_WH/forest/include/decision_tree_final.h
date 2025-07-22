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
  void buildFlatTree(const json& treeData);
  void buildFlatRecursive(const json& treeData, int treeIdx, int flatIdx, int maxDepth, int currentDepth);
  int computeDepth(const json& treeData, int nodeIdx, int currentDepth) const;
  void fillDummyLeaf(int idx, int predictedClass);
  std::vector<float> getFeatures() const;
  std::vector<float> getValues() const;
  std::vector<float> getTreshold() const;

private:
  std::vector<int> features;
  std::vector<float> thresholds;
  std::vector<int> values;
  std::vector<std::string> classLabels;
  int depth = 0;

};