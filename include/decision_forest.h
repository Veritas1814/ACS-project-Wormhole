#pragma once

#include "decision_tree_final.h"
#include <vector>
#include <string>
#include <map>

class RandomForest {
public:
    std::vector<DecisionTreeFinal> trees;
    std::vector<std::string> classLabels;

    void loadFromJson(const std::string& filename);
    std::pair<std::vector<int>, std::string> predict(const std::vector<double>& sample);
};
