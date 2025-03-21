#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include "modified/decision_tree_op1.cpp"  // include the modified decision tree versions
#include "modified/decision_tree_op2.cpp"
#include "modified/decision_tree_op3.cpp"
#include "modified/decision_tree_op4.cpp"
#include "modified/decision_tree_op5.cpp"
#include "modified/decision_tree_op6.cpp"
#include "../include/decision_tree.h"


using json = nlohmann::json;

template<typename T>
void readCSV(const std::string& filename, std::vector<std::vector<T>>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<T> row;
        std::string value;
        while (std::getline(ss, value, ',')) {
            row.push_back(static_cast<T>(std::stod(value)));
        }
        data.push_back(row);
    }
}

template<typename TreeType>
void loadTreeFromJson(TreeType& tree, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open JSON file: " << filename << std::endl;
        return;
    }

    json treeData;
    try {
        file >> treeData;
    } catch (const json::parse_error& e) {
        std::cerr << "JSON Parse Error: " << e.what() << std::endl;
        return;
    }

    std::cout << "Loaded JSON Data: " << treeData.dump(4) << std::endl;

    try {
        tree.loadFromJson(treeData.dump());  // Ensure correct format
    } catch (const std::exception& e) {
        std::cerr << "Error loading JSON into tree: " << e.what() << std::endl;
    }
}

#define BENCHMARK_DECISION_TREE(NAME, TREE_CLASS, DATA_TYPE) \
static void NAME(benchmark::State& state) { \
    TREE_CLASS tree; \
    tree.loadFromJson("../data/tree.json"); \
    std::vector<std::vector<DATA_TYPE>> testData = {{5.1, 3.5, 1.4, 0.2}}; \
    for (auto _ : state) { \
        for (const auto& sample : testData) { \
            tree.predict(sample); \
            /*std::cout << "Predicted: " << tree.predict(sample) << "\n";*/\
        } \
    } \
} \
BENCHMARK(NAME);

BENCHMARK_DECISION_TREE(BM_DecisionTree_Original, DecisionTree, double)
BENCHMARK_DECISION_TREE(BM_DecisionTree_Op1, DecisionTreeOp1, double)
BENCHMARK_DECISION_TREE(BM_DecisionTree_Op2, DecisionTreeOp2, float)
BENCHMARK_DECISION_TREE(BM_DecisionTree_Op3, DecisionTreeOp3, double)
BENCHMARK_DECISION_TREE(BM_DecisionTree_Op4, DecisionTreeOp4, double)
BENCHMARK_DECISION_TREE(BM_DecisionTree_Op5, DecisionTreeOp5, double)
BENCHMARK_DECISION_TREE(BM_DecisionTree_Op6, DecisionTreeOp6, double)

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}



//#include <benchmark/benchmark.h>
//#include <iostream>
//#include <vector>
//#include <string>
//#include <fstream>
//#include <sstream>
//#include <nlohmann/json.hpp>
//#include <memory>
//
//using json = nlohmann::json;
//
//struct Node {
//    bool isLeaf = false;
//    int feature = -1;
//    double threshold = 0.0;
//    int value = -1;
//    std::shared_ptr<Node> left = nullptr;
//    std::shared_ptr<Node> right = nullptr;
//};
//
//class DecisionTree {
//public:
//    void loadFromJson(const std::string& filename);
//    std::string predict(const std::vector<double>& sample);
//private:
//    std::shared_ptr<Node> buildTree(const json& treeData, int index);
//    void loadTree(const json& treeData);
//    std::shared_ptr<Node> root;
//    std::vector<std::string> classLabels;
//};
//
//void DecisionTree::loadFromJson(const std::string& filename) {
//    std::ifstream file(filename);
//    if (!file.is_open()) {
//        std::cerr << "Error: Cannot open file " << filename << std::endl;
//        return;
//    }
//
//    json treeData;
//    try {
//        file >> treeData;
//    } catch (const json::parse_error& e) {
//        std::cerr << "JSON Parse Error: " << e.what() << std::endl;
//        return;
//    }
//
//    if (!treeData.contains("tree") || !treeData["tree"].is_object()) {
//        std::cerr << "Error: JSON does not contain a valid 'tree' object." << std::endl;
//        return;
//    }
//
//    auto tree = treeData["tree"];
//    if (!tree.contains("classes") || !tree["classes"].is_array()) {
//        std::cerr << "Error: JSON does not contain 'classes' array." << std::endl;
//        return;
//    }
//
//    classLabels = tree["classes"].get<std::vector<std::string>>();
//
//    try {
//        loadTree(tree);
//    } catch (const std::exception& e) {
//        std::cerr << "Error loading tree: " << e.what() << std::endl;
//    }
//}
//
//std::shared_ptr<Node> DecisionTree::buildTree(const json& treeData, int index) {
//    if (index < 0 || index >= treeData["children_left"].size()) {
//        std::cerr << "Error: Invalid tree index " << index << std::endl;
//        return nullptr;
//    }
//
//    auto node = std::make_shared<Node>();
//
//    if (treeData["children_left"][index] == -1) {
//        node->isLeaf = true;
//        if (!treeData.contains("value") || !treeData["value"].is_array()) {
//            std::cerr << "Error: Missing 'value' array in JSON." << std::endl;
//            return nullptr;
//        }
//        const auto& values = treeData["value"][index][0];
//        if (values.empty()) {
//            std::cerr << "Error: Empty value array for leaf node." << std::endl;
//            return nullptr;
//        }
//        node->value = std::distance(values.begin(), std::max_element(values.begin(), values.end()));
//    } else {
//        if (!treeData.contains("feature") || !treeData.contains("threshold")) {
//            std::cerr << "Error: Missing 'feature' or 'threshold' keys in JSON." << std::endl;
//            return nullptr;
//        }
//        node->feature = treeData["feature"][index];
//        node->threshold = treeData["threshold"][index];
//        node->left = buildTree(treeData, treeData["children_left"][index]);
//        node->right = buildTree(treeData, treeData["children_right"][index]);
//        if (!node->left || !node->right) {
//            std::cerr << "Error: One of the child nodes is null." << std::endl;
//            return nullptr;
//        }
//    }
//    return node;
//}
//
//void DecisionTree::loadTree(const json& treeData) {
//    root = buildTree(treeData, 0);
//}
//
//std::string DecisionTree::predict(const std::vector<double>& sample) {
//    if (!root) {
//        std::cerr << "Error: DecisionTree root is null!" << std::endl;
//        return "";
//    }
//    auto node = root;
//    while (node && !node->isLeaf) {
//        if (node->feature < 0 || node->feature >= sample.size()) {
//            std::cerr << "Error: Invalid feature index " << node->feature << std::endl;
//            return "";
//        }
//        node = (sample[node->feature] < node->threshold) ? node->left : node->right;
//        if (!node) {
//            std::cerr << "Error: Traversed to a null node!" << std::endl;
//            return "";
//        }
//    }
//    if (!node) {
//        std::cerr << "Error: Final node is null!" << std::endl;
//        return "";
//    }
//    if (node->value < 0 || node->value >= classLabels.size()) {
//        std::cerr << "Error: Invalid class label index " << node->value << std::endl;
//        return "";
//    }
//    return classLabels[node->value];
//}
//
//#define BENCHMARK_DECISION_TREE(NAME, TREE_CLASS, DATA_TYPE) \
//static void NAME(benchmark::State& state) { \
//    TREE_CLASS tree; \
//    tree.loadFromJson("../data/tree.json"); \
//    std::vector<std::vector<DATA_TYPE>> testData = {{5.1, 3.5, 1.4, 0.2}}; \
//    for (auto _ : state) { \
//        for (const auto& sample : testData) { \
//            tree.predict(sample); \
//        } \
//    } \
//} \
//BENCHMARK(NAME);
//
//BENCHMARK_DECISION_TREE(BM_DecisionTree_Original, DecisionTree, double)
//
//int main(int argc, char** argv) {
//    ::benchmark::Initialize(&argc, argv);
//    ::benchmark::RunSpecifiedBenchmarks();
//    return 0;
//}
