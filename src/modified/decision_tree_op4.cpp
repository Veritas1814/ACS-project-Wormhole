// decision_tree_op4.cpp
#include "../include/decision_tree.h"
#include <fstream>
#include <iostream>
#include <queue>
#include <vector>
#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct NodeOp4 {
    int feature;
    double threshold;
    int value;
    bool isLeaf;
    int leftIndex;
    int rightIndex;
};

class DecisionTreeOp4 {
public:
    std::vector<NodeOp4> nodes;
    std::vector<std::string> classLabels;
    
    // Build the tree in level order.
    void loadFromJson(const std::string& filename) {
        std::ifstream file(filename);
        if(!file.is_open()){
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }
        json treeData;
        file >> treeData;
        auto tree = treeData["tree"];
        classLabels = tree["classes"].get<std::vector<std::string>>();
        nodes.clear();
        
        // Use a queue to hold JSON indices that need processing.
        std::queue<int> q;
        q.push(0);
        while (!q.empty()) {
            int idx = q.front();
            q.pop();
            NodeOp4 node;
            if (tree["children_left"][idx] == -1) {
                node.isLeaf = true;
                const auto& vals = tree["value"][idx][0];
                int maxIdx = 0;
                for (int i = 1; i < vals.size(); ++i) {
                    if (vals[i] > vals[maxIdx])
                        maxIdx = i;
                }
                node.value = maxIdx;
                node.feature = -1;
                node.threshold = 0.0;
                node.leftIndex = -1;
                node.rightIndex = -1;
            } else {
                node.isLeaf = false;
                node.feature = tree["feature"][idx];
                node.threshold = tree["threshold"][idx];
                // Enqueue children for level-order processing.
                q.push(tree["children_left"][idx]);
                q.push(tree["children_right"][idx]);
                // For demonstration, store the raw JSON child indices.
                node.leftIndex = tree["children_left"][idx];
                node.rightIndex = tree["children_right"][idx];
            }
            nodes.push_back(node);
        }
    }
    
    std::string predict(const std::vector<double>& sample) {
        int cur = 0;
        while (!nodes[cur].isLeaf) {
            cur = (sample[nodes[cur].feature] < nodes[cur].threshold)
                      ? nodes[cur].leftIndex
                      : nodes[cur].rightIndex;
        }
        return classLabels[nodes[cur].value];
    }
};