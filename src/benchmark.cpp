#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <nlohmann/json.hpp>

#include "decision_tree_op1.h"
#include "decision_tree_op2.h"
#include "decision_tree_op3.h"
#include "decision_tree_op4.h"
#include "decision_tree_final.h"
#include "decision_tree.h"

using json = nlohmann::json;

#ifdef __linux__
void clearCache() {
    system("sync");
    system("echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null");
}
#else
void clearCache() {}
#endif

template<typename T>
void readCSV(const std::string& filename, std::vector<std::vector<T>>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    std::string line;
    std::getline(file, line);
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

template<typename TreeType, typename T>
static void BM_DecisionTree_Full(benchmark::State& state, const std::string& treeJson, const std::string& testCsv) {
    TreeType tree;
    tree.loadFromJson(treeJson);

    std::vector<std::vector<T>> testData;
    readCSV(testCsv, testData);

    clearCache();

    for (auto _ : state) {
        for (const auto& sample : testData) {
            auto pred = tree.predict(sample);
            benchmark::DoNotOptimize(pred);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <tree_json> <test_csv>" << std::endl;
        return 1;
    }

    std::string treeJson = std::string("../data/") + argv[1];
    std::string testCsv  = std::string("../data/") + argv[2];

    benchmark::RegisterBenchmark("BM_DecisionTree_Original", [=](benchmark::State& state) {
        BM_DecisionTree_Full<DecisionTree, double>(state, treeJson, testCsv);
    });
    benchmark::RegisterBenchmark("BM_DecisionTree_Op1", [=](benchmark::State& state) {
        BM_DecisionTree_Full<DecisionTreeOp1, double>(state, treeJson, testCsv);
    });
    benchmark::RegisterBenchmark("BM_DecisionTree_Op2", [=](benchmark::State& state) {
        BM_DecisionTree_Full<DecisionTreeOp2, float>(state, treeJson, testCsv);
    });
    benchmark::RegisterBenchmark("BM_DecisionTree_Op3", [=](benchmark::State& state) {
        BM_DecisionTree_Full<DecisionTreeOp3, double>(state, treeJson, testCsv);
    });
    benchmark::RegisterBenchmark("BM_DecisionTree_Op4", [=](benchmark::State& state) {
        BM_DecisionTree_Full<DecisionTreeOp4, double>(state, treeJson, testCsv);
    });
    benchmark::RegisterBenchmark("BM_DecisionTree_Final", [=](benchmark::State& state) {
        BM_DecisionTree_Full<DecisionTreeFinal, float>(state, treeJson, testCsv);
    });

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
