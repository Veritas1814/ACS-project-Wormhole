#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

#include "decision_tree_op1.h"  // include the modified decision tree versions
#include "decision_tree_op2.h"
#include "decision_tree_op3.h"
#include "decision_tree_op4.h"
#include "decision_tree_op5.h"
#include "decision_tree_op6.h"

#include "decision_tree.h"

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

void saveCSV(const std::string& filename, const std::vector<std::string>& predictions) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    file << "Predicted_Species" << std::endl;
    for (const auto& pred : predictions) {
        file << pred << std::endl;
    }
}

void readPredictionsCSV(const std::string& filename, std::vector<std::string>& predictions) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        predictions.push_back(line);
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

void comparePredictions(const std::vector<std::string>& py_preds, const std::vector<std::string>& cpp_preds) {
    std::cout << "Comparison done" << std::endl;

    int failed_count = 0;
    for (size_t i = 0; i < py_preds.size(); ++i) {
        if (py_preds[i] != cpp_preds[i]) {
            ++failed_count;
            std::cout << "Failed: " << i << ", Predicted: " << cpp_preds[i] << ", Actual: " << py_preds[i] << std::endl;
        }
    }

    if (failed_count == 0) {
        std::cout << "All correct" << std::endl;
    } else {
        std::cout << failed_count << " failed classes" << std::endl;
    }
}

template<typename TreeType, typename T>
void BM_DecisionTree(benchmark::State& state, const std::string& treeJson, const std::string& testCsv, const std::string& pyPredictionsCsv, const std::string& cppPredictionsCsv, const std::string& suffix) {
    TreeType tree;
    tree.loadFromJson(treeJson);

    std::vector<std::vector<T>> testData;
    readCSV(testCsv, testData);

    std::vector<std::string> cpp_predictions;

    for (auto _ : state) {
        for (const auto& sample : testData) {
            cpp_predictions.push_back(tree.predict(sample));
        }
    }

    // After benchmarking loop is done, compare predictions
    std::vector<std::string> py_predictions;
    readPredictionsCSV(pyPredictionsCsv, py_predictions);

    comparePredictions(py_predictions, cpp_predictions);

    // Save predictions to a CSV
    std::string outputCsv = cppPredictionsCsv.substr(0, cppPredictionsCsv.find_last_of('.')) + suffix;
    saveCSV(outputCsv, cpp_predictions);
    std::cout << "Predictions saved to " << outputCsv << std::endl;
}



int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <tree_json> <test_csv> <python_predictions_csv> <cpp_predictions_csv>" << std::endl;
        return 1;
    }

    // Get file paths from arguments
    std::string treeJson = std::string("../data/") + argv[1];
    std::string testCsv = std::string("../data/") + argv[2];
    std::string pyPredictionsCsv = std::string("../data/") + argv[3];
    std::string cppPredictionsCsv = std::string("../data/") + argv[4];

    benchmark::RegisterBenchmark("BM_DecisionTree_Original", [=](benchmark::State& state) {
       BM_DecisionTree<DecisionTree, double>(state, treeJson, testCsv, pyPredictionsCsv, cppPredictionsCsv, "_original.csv");
    });
    benchmark::RegisterBenchmark("BM_DecisionTree_Op1", [=](benchmark::State& state) {
       BM_DecisionTree<DecisionTreeOp1, double>(state, treeJson, testCsv, pyPredictionsCsv, cppPredictionsCsv, "_op1.csv");
    });
    benchmark::RegisterBenchmark("BM_DecisionTree_Op2", [=](benchmark::State& state) {
       BM_DecisionTree<DecisionTreeOp2, float>(state, treeJson, testCsv, pyPredictionsCsv, cppPredictionsCsv, "_op2.csv");
    });
    benchmark::RegisterBenchmark("BM_DecisionTree_Op3", [=](benchmark::State& state) {
       BM_DecisionTree<DecisionTreeOp3, double>(state, treeJson, testCsv, pyPredictionsCsv, cppPredictionsCsv, "_op3.csv");
    });
    benchmark::RegisterBenchmark("BM_DecisionTree_Op4", [=](benchmark::State& state) {
       BM_DecisionTree<DecisionTreeOp4, double>(state, treeJson, testCsv, pyPredictionsCsv, cppPredictionsCsv, "_op4.csv");
    });
    benchmark::RegisterBenchmark("BM_DecisionTree_Op5", [=](benchmark::State& state) {
       BM_DecisionTree<DecisionTreeOp5, double>(state, treeJson, testCsv, pyPredictionsCsv, cppPredictionsCsv, "_op5.csv");
    });
    benchmark::RegisterBenchmark("BM_DecisionTree_Op6", [=](benchmark::State& state) {
       BM_DecisionTree<DecisionTreeOp6, double>(state, treeJson, testCsv, pyPredictionsCsv, cppPredictionsCsv, "_op6.csv");
    });

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
