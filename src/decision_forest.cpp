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
    try {
        file >> forestData;
    } catch (const std::exception &e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return;
    }

    auto& forest = forestData["forest"];
    if (!forest.contains("classes")) {
        std::cerr << "Error: Forest JSON does not contain 'classes'" << std::endl;
        return;
    }
    classLabels = forest["classes"].get<std::vector<std::string>>();

    if (forest["feature"].size() != forest["threshold"].size() ||
        forest["children_left"].size() != forest["feature"].size() ||
        forest["children_right"].size() != forest["feature"].size() ||
        forest["value"].size() != forest["feature"].size()) {
        std::cerr << "Error: Mismatch in forest array sizes" << std::endl;
        return;
    }

    trees.clear();
    for (size_t i = 0; i < forest["feature"].size(); i++) {
        DecisionTree tree;
        json treeJson;
        treeJson["feature"]        = forest["feature"][i];
        treeJson["threshold"]      = forest["threshold"][i];
        treeJson["children_left"]  = forest["children_left"][i];
        treeJson["children_right"] = forest["children_right"][i];
        treeJson["value"]          = forest["value"][i];

        treeJson["classes"] = forest["classes"];

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
            std::string predStr = tree.predict(sample);

            // Find the index of the predicted class
            auto it = std::find(classLabels.begin(), classLabels.end(), predStr);
            if (it == classLabels.end()) {
                std::cerr << "Error: Class label '" << predStr << "' not found in classLabels" << std::endl;
                continue;
            }
            prediction = std::distance(classLabels.begin(), it);
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

    if (votes.empty()) {
        std::cerr << "Error: No valid votes in random forest prediction" << std::endl;
        return {std::vector<int>(classLabels.size(), 0), ""}; // Return zero votes and empty label
    }

    auto maxVote = std::max_element(votes.begin(), votes.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    if (maxVote == votes.end()) {
        std::cerr << "Error: No valid votes in random forest prediction" << std::endl;
        return {std::vector<int>(classLabels.size(), 0), ""};
    }

    std::vector<int> voteCounts(classLabels.size(), 0);
    for (const auto& [index, count] : votes) {
        voteCounts[index] = count;
    }

    return {voteCounts, classLabels[maxVote->first]};
}
