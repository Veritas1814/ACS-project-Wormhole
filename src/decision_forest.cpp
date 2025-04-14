#include "decision_forest.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

void RandomForest::loadFromJson(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    json forestData;

    file >> forestData;


    auto& forest = forestData["forest"];
    if (!forest.contains("classes")) {
        std::cerr << "Error: Forest JSON does not contain 'classes'" << std::endl;
        return;
    }

    classLabels = forest["classes"].get<std::vector<std::string>>();  // Optional: for metadata

    trees.clear();
    for (size_t i = 0; i < forest["feature"].size(); i++) {
        DecisionTree tree;
        json treeJson;
        treeJson["feature"]        = forest["feature"][i];
        treeJson["threshold"]      = forest["threshold"][i];
        treeJson["children_left"]  = forest["children_left"][i];
        treeJson["children_right"] = forest["children_right"][i];
        treeJson["value"]          = forest["value"][i];

        treeJson["classes"]        = forest["classes"];

        tree.loadTree(treeJson);

        trees.push_back(tree);
    }
}

std::pair<std::vector<int>, int> RandomForest::predict(const std::vector<double>& sample) noexcept {
    size_t numClasses = classLabels.size();  // Could be passed in from elsewhere if classLabels is removed
    std::vector<int> voteCounts(numClasses, 0);

    for (auto& tree : trees) {
        int classIndex = tree.predict(sample);  // <<< This must return an int index
        if (classIndex >= 0 && static_cast<size_t>(classIndex) < numClasses) {
            voteCounts[classIndex]++;
        } else {
            std::cerr << "Warning: Invalid class index " << classIndex << " from tree" << std::endl;
        }
    }

    // Get class with max votes
    int predictedClass = std::distance(voteCounts.begin(),
                           std::max_element(voteCounts.begin(), voteCounts.end()));

    return {voteCounts, predictedClass};
}
