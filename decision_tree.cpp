#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

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
    std::vector<std::string> classLabels;

    void loadFromJson(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }

        json treeData;
        file >> treeData;
        auto tree = treeData["tree"];
        classLabels = tree["classes"].get<std::vector<std::string>>();
        root = buildTree(tree, 0);
    }

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

    std::string predict(const std::vector<double>& sample) {
        auto node = root;
        while (!node->isLeaf) {
            node = (sample[node->feature] < node->threshold) ? node->left : node->right;
        }
        return classLabels[node->value];
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

void saveCSV(const std::string& filename, const std::vector<std::string>& predictions) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    file << "Prediction" << std::endl;
    for (const auto& pred : predictions) {
        file << pred << std::endl;
    }
}

int main() {
    DecisionTree tree;
    tree.loadFromJson("tree.json");

    std::vector<std::vector<double>> testData;
    readCSV("iris_test.csv", testData);

    std::vector<std::string> predictions;
    for (const auto& sample : testData) {
        predictions.push_back(tree.predict(sample));
    }

    saveCSV("iris_test_result.csv", predictions);
    return 0;
}
