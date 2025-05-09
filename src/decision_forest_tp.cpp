#include "decision_forest.h"
#include "thread_pool.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <future>
#include <map>
#include <vector>
#include <algorithm>

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
        treeJson["classes"]        = forest["classes"];

        tree.loadTree(treeJson);

        trees.push_back(tree);
    }
}

std::pair<std::vector<int>, int> RandomForest::predict(const std::vector<double>& sample) noexcept {
    ThreadPool pool(std::thread::hardware_concurrency());

    std::map<int, int> votes;
    std::mutex voteMutex;
    std::vector<std::future<int>> futures;

    for (auto& tree : trees) {
        futures.push_back(pool.submit([&tree, &sample]() -> int {
            return tree.predict(sample);
        }));
    }

    for (auto& future : futures) {
        int prediction = future.get();
        if (prediction >= 0 && static_cast<size_t>(prediction) < classLabels.size()) {
            std::lock_guard<std::mutex> lock(voteMutex);
            votes[prediction]++;
        }
    }

    if (votes.empty()) {
        std::cerr << "Error: No valid votes in random forest prediction" << std::endl;
        return {std::vector<int>(classLabels.size(), 0), -1};
    }

    auto maxVote = std::max_element(votes.begin(), votes.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<int> voteCounts(classLabels.size(), 0);
    for (const auto& [classIdx, count] : votes) {
        voteCounts[classIdx] = count;
    }

    return {voteCounts, maxVote->first};
}
