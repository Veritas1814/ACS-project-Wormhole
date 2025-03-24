// decision_tree_op3.cpp
#include "../include/decision_tree.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class DecisionTreeOp3 {
public:
    std::vector<int> features;
    std::vector<double> thresholds;
    std::vector<int> values;
    std::vector<bool> isLeaf;
    std::vector<int> leftIndices;
    std::vector<int> rightIndices;
    std::vector<std::string> classLabels;
    
    // Build tree recursively and store each field in a separate vector.
    int buildTree(const json& treeData, int index) {
        int curIndex = features.size();
        features.push_back(-1);
        thresholds.push_back(0.0);
        values.push_back(-1);
        isLeaf.push_back(false);
        leftIndices.push_back(-1);
        rightIndices.push_back(-1);
        
        if (treeData["children_left"][index] == -1) {
            isLeaf[curIndex] = true;
            const auto& vals = treeData["value"][index][0];
            int maxIdx = 0;
            for (int i = 1; i < vals.size(); ++i) {
                if (vals[i] > vals[maxIdx])
                    maxIdx = i;
            }
            values[curIndex] = maxIdx;
        } else {
            features[curIndex] = treeData["feature"][index];
            thresholds[curIndex] = treeData["threshold"][index];
            int left = buildTree(treeData, treeData["children_left"][index]);
            int right = buildTree(treeData, treeData["children_right"][index]);
            leftIndices[curIndex] = left;
            rightIndices[curIndex] = right;
        }
        return curIndex;
    }
    
    void loadFromJson(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()){
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }
        json treeData;
        file >> treeData;
        auto tree = treeData["tree"];
        classLabels = tree["classes"].get<std::vector<std::string>>();
        features.clear();
        thresholds.clear();
        values.clear();
        isLeaf.clear();
        leftIndices.clear();
        rightIndices.clear();
        buildTree(tree, 0);
    }
    
    std::string predict(const std::vector<double>& sample) {
        int cur = 0;
        while (!isLeaf[cur]) {
            cur = (sample[features[cur]] < thresholds[cur]) ? leftIndices[cur] : rightIndices[cur];
        }
        return classLabels[values[cur]];
    }
};