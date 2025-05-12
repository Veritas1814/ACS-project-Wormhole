#pragma once

#include "decision_tree_final.h"
#include <vector>
#include <string>
#include <map>

class RandomForest {
public:
    std::vector<DecisionTreeFinal> trees;
    std::vector<std::string> classLabels;
    std::vector<float> flattenForest() const;
    void loadFromJson(const std::string& filename);
    std::pair<std::vector<int>, int> predict(const std::vector<float>& sample);
};
