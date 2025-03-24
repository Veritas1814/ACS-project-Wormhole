#include "../include/decision_forest.h"
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
    try {
        file >> forestData;
    } catch (const std::exception &e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return;
    }

    // Debug print to ensure the file loaded correctly
    std::cout << "Loaded forest data: " << forestData.dump(4) << std::endl;

    auto& forest = forestData["forest"];
    if (!forest.contains("classes")) {
        std::cerr << "Error: Forest JSON does not contain 'classes'" << std::endl;
        return;
    }
    classLabels = forest["classes"].get<std::vector<std::string>>();

    std::cout << "Class labels: ";
    for (const auto& label : classLabels) {
        std::cout << label << " ";
    }
    std::cout << std::endl;

    // Validate that all arrays have the same number of trees
    size_t numTrees = forest["feature"].size();
    if (forest["threshold"].size() != numTrees ||
        forest["children_left"].size() != numTrees ||
        forest["children_right"].size() != numTrees ||
        forest["value"].size() != numTrees) {
        std::cerr << "Error: Mismatch in forest array sizes" << std::endl;
        return;
    }
    std::cout << "Number of trees in the forest: " << numTrees << std::endl;

    trees.clear();
    for (size_t i = 0; i < numTrees; i++) {
        DecisionTree tree;
        json treeJson;
        // Construct tree JSON for this individual tree.
        treeJson["feature"]        = forest["feature"][i];
        treeJson["threshold"]      = forest["threshold"][i];
        treeJson["children_left"]  = forest["children_left"][i];
        treeJson["children_right"] = forest["children_right"][i];
        treeJson["value"]          = forest["value"][i];

        try {
            tree.loadTree(treeJson);
        } catch (const std::exception &e) {
            std::cerr << "Error loading tree " << i << ": " << e.what() << std::endl;
            continue;
        }
        trees.push_back(tree);
    }
}

std::pair<std::vector<int>, std::string> RandomForest::predict(const std::vector<double>& sample) {
    std::map<int, int> votes;
    for (size_t i = 0; i < classLabels.size(); i++) {
        votes[i] = 0;
    }

    for (auto& tree : trees) {
        int prediction = -1;
        try {
            // Assuming tree.predict returns a string convertible to int
            std::string predStr = tree.predict(sample);
            prediction = std::stoi(predStr);
        } catch (const std::exception &e) {
            std::cerr << "Error in tree prediction: " << e.what() << std::endl;
            continue;
        }
        if (prediction < 0 || static_cast<size_t>(prediction) >= classLabels.size()) {
            std::cerr << "Warning: tree prediction " << prediction << " is out of valid range" << std::endl;
            continue;
        }
        votes[prediction]++;
    }

    auto maxVote = std::max_element(votes.begin(), votes.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    if (maxVote == votes.end()) {
        throw std::runtime_error("No valid votes in random forest prediction");
    }

    std::vector<int> voteCounts;
    for (size_t i = 0; i < classLabels.size(); i++) {
        voteCounts.push_back(votes[i]);
    }

    return {voteCounts, classLabels[maxVote->first]};
}