#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "decision_forest.h"  // Assuming this includes the RandomForest class.
#include <cstdlib>

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

void readPredictionsCSV(const std::string& filename, std::vector<int>& predictions) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        predictions.push_back(std::stoi(line));
    }
}

void saveCSV(const std::string& filename,
             const std::vector<std::vector<int>>& voteStats,
             const std::vector<int>& finalPredictions,
             int numClasses) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open output file " << filename << std::endl;
        return;
    }

    // Write header
    for (int i = 0; i < numClasses; ++i) {
        file << i << ",";
    }
    file << "prediction\n";

    // Write vote stats and predictions
    for (size_t i = 0; i < voteStats.size(); ++i) {
        for (int count : voteStats[i]) {
            file << count << ",";
        }
        file << finalPredictions[i] << "\n";
    }
}

void comparePredictions(const std::vector<int>& py_preds, const std::vector<int>& cpp_preds) {
    if (py_preds.size() != cpp_preds.size()) {
        std::cerr << "Prediction size mismatch!" << std::endl;
        return;
    }

    int failed_count = 0;
    for (size_t i = 0; i < py_preds.size(); ++i) {
        if (py_preds[i] != cpp_preds[i]) {
            ++failed_count;
            std::cout << "Mismatch at index " << i << ": Python = "
                      << py_preds[i] << ", C++ = " << cpp_preds[i] << std::endl;
        }
    }

    if (failed_count == 0) {
        std::cout << "All predictions match!" << std::endl;
    } else {
        std::cout << failed_count << " mismatches found." << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <tree_json> <test_csv> <python_predictions_csv> <cpp_predictions_csv>" << std::endl;
        return 1;
    }

    std::string treeJson = std::string("../data/") + argv[1];
    std::string testCsv = std::string("../data/") + argv[2];
    std::string pyPredictionsCsv = std::string("../data/") + argv[3];
    std::string cppPredictionsCsv = std::string("../data/") + argv[4];

    RandomForest forest;
    forest.loadFromJson(treeJson);

    std::vector<std::vector<double>> testData;
    readCSV(testCsv, testData);

    std::vector<std::vector<int>> voteStats;
    std::vector<int> finalPredictions;

    for (const auto& sample : testData) {
        auto [votes, prediction] = forest.predict(sample);
        voteStats.push_back(votes);
        finalPredictions.push_back(prediction);  // Ensure prediction is numeric
    }

    std::vector<int> py_predictions;
    readPredictionsCSV(pyPredictionsCsv, py_predictions);

    comparePredictions(py_predictions, finalPredictions);

    int numClasses = voteStats.empty() ? 0 : voteStats[0].size();
    saveCSV(cppPredictionsCsv, voteStats, finalPredictions, numClasses);

    std::cout << "Test completed. Results saved to: " << cppPredictionsCsv << std::endl;

    return 0;
}
