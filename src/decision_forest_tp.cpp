#include "decision_forest.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <future>
#include <map>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>

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
        DecisionTreeFinal tree;
        json treeJson;
        treeJson["feature"]        = forest["feature"][i];
        treeJson["threshold"]      = forest["threshold"][i];
        treeJson["children_left"]  = forest["children_left"][i];
        treeJson["children_right"] = forest["children_right"][i];
        treeJson["value"]          = forest["value"][i];
        treeJson["classes"]        = forest["classes"];

        try {
            tree.buildTree(treeJson);
        } catch (const std::exception &e) {
            std::cerr << "Error loading tree " << i << ": " << e.what() << std::endl;
            continue;
        }
        trees.push_back(tree);
    }
}
std::pair<std::vector<int>, std::string> RandomForest::predict(const std::vector<double>& sample) {
    size_t num_threads = std::thread::hardware_concurrency();

    std::vector<std::thread> threads;
    std::mutex voteMutex;
    std::map<int, int> votes;
    std::vector<int> voteCounts(classLabels.size(), 0);

    size_t trees_per_thread = (trees.size() + num_threads - 1) / num_threads;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start_index = i * trees_per_thread;
        size_t end_index = std::min(start_index + trees_per_thread, trees.size());

        threads.emplace_back([&, start_index, end_index]() {
            std::map<int, int> localVotes;
            for (size_t j = start_index; j < end_index; ++j) {
                try {
                    std::cout << "Tree #" << j << ": predicting...\n";
                    std::string predStr = trees[j].predict(sample);
                    std::cout << "Tree #" << j << ": prediction = " << predStr << "\n";
                    auto it = std::find(classLabels.begin(), classLabels.end(), predStr);
                    if (it != classLabels.end()) {
                        int classIdx = std::distance(classLabels.begin(), it);
                        localVotes[classIdx]++;
                    }
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(voteMutex);
                    std::cerr << "Exception in tree " << j << ": " << e.what() << std::endl;
                }
            }
            std::lock_guard<std::mutex> lock(voteMutex);
            for (const auto& [idx, cnt] : localVotes) {
                votes[idx] += cnt;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    if (votes.empty()) {
        std::cerr << "Error: No valid votes in random forest prediction" << std::endl;
        return {std::vector<int>(classLabels.size(), 0), ""};
    }

    for (const auto& [classIdx, count] : votes) {
        voteCounts[classIdx] = count;
    }

    auto maxVote = std::max_element(votes.begin(), votes.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    return {voteCounts, classLabels[maxVote->first]};
}
