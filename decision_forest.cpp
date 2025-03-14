#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <map>

using json = nlohmann::json;

class Node {
public:
    int feature;
    double threshold;
    int value;
    bool isLeaf;
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;

    Node() : feature(-1), threshold(0.0), value(-1), isLeaf(false), left(nullptr), right(nullptr) {}
};

class DecisionTree {
public:
    std::shared_ptr<Node> root;

    std::shared_ptr<Node> buildTree(const json& treeData, int index) {
        auto node = std::make_shared<Node>();
        if (treeData["children_left"][index] == -1) {
            node->isLeaf = true;
            const auto& values = treeData["value"][index][0];
            node->value = std::distance(values.begin(), std::max_element(values.begin(), values.end()));
        } else {
            node->feature = treeData["feature"][index];
            node->threshold = treeData["threshold"][index];
            node->left = buildTree(treeData, treeData["children_left"][index]);
            node->right = buildTree(treeData, treeData["children_right"][index]);
        }
        return node;
    }

    void loadTree(const json& treeData) {
        root = buildTree(treeData, 0);
    }

    int predict(const std::vector<double>& sample) {
        auto node = root;
        while (!node->isLeaf) {
            node = (sample[node->feature] < node->threshold) ? node->left : node->right;
        }
        return node->value;
    }
};

class RandomForest {
public:
    std::vector<DecisionTree> trees;
    std::vector<std::string> classLabels;

    void loadFromJson(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }

        json forestData;
        file >> forestData;
        auto& forest = forestData["forest"];
        classLabels = forest["classes"].get<std::vector<std::string>>();

        for (size_t i = 0; i < forest["feature"].size(); i++) {
            DecisionTree tree;
            json treeJson = {
                {"feature", forest["feature"][i]},
                {"threshold", forest["threshold"][i]},
                {"children_left", forest["children_left"][i]},
                {"children_right", forest["children_right"][i]},
                {"value", forest["value"][i]}
            };
            tree.loadTree(treeJson);
            trees.push_back(tree);
        }
    }

    std::pair<std::vector<int>, std::string> predict(const std::vector<double>& sample) {
        std::map<int, int> votes;
        for (size_t i = 0; i < classLabels.size(); i++) {
            votes[i] = 0;
        }

        for (auto& tree : trees) {
            int prediction = tree.predict(sample);
            votes[prediction]++;
        }

        auto maxVote = std::max_element(votes.begin(), votes.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        std::vector<int> voteCounts;
        for (size_t i = 0; i < classLabels.size(); i++) {
            voteCounts.push_back(votes[i]);
        }

        return {voteCounts, classLabels[maxVote->first]};
    }
};

void readCSV(const std::string& filename, std::vector<std::vector<double>>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string value;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }
        data.push_back(row);
    }
}

void saveCSV(const std::string& filename, const std::vector<std::vector<int>>& voteStats, const std::vector<std::string>& finalPredictions, const std::vector<std::string>& classLabels) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    for (const auto& label : classLabels) {
        file << label << ",";
    }
    file << "prediction\n";

    for (size_t i = 0; i < voteStats.size(); i++) {
        for (const auto& count : voteStats[i]) {
            file << count << ",";
        }
        file << finalPredictions[i] << "\n";
    }
}

int main() {
    RandomForest forest;
    forest.loadFromJson("forest.json");

    std::vector<std::vector<double>> testData;
    readCSV("iris_test.csv", testData);

    std::vector<std::vector<int>> voteStats;
    std::vector<std::string> finalPredictions;

    for (const auto& sample : testData) {
        auto [votes, finalPrediction] = forest.predict(sample);
        voteStats.push_back(votes);
        finalPredictions.push_back(finalPrediction);
    }

    saveCSV("iris_test_votes.csv", voteStats, finalPredictions, forest.classLabels);

    return 0;
}
