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

void saveCSV(const std::string& filename,
             const std::vector<std::vector<int>>& voteStats,
             const std::vector<std::string>& finalPredictions,
             const std::vector<std::string>& classLabels) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open output file " << filename << std::endl;
        return;
    }

    // Write header row.
    for (const auto& label : classLabels) {
        file << label << ",";
    }
    file << "prediction" << "\n";

    // Write vote counts and prediction for each sample.
    for (size_t i = 0; i < voteStats.size(); i++) {
        for (const auto& count : voteStats[i]) {
            file << count << ",";
        }
        file << finalPredictions[i] << "\n";
    }
}

void comparePredictions(const std::vector<std::string>& py_preds, const std::vector<std::string>& cpp_preds) {
    std::cout << "comparison done" << std::endl;

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

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <forest_json> <test_csv> <python_predictions_csv> <cpp_predictions_csv>" << std::endl;
        return 1;
    }

    // Get file paths from arguments
    std::string forestJson = std::string("../data/") + argv[1];
    std::string testCsv = std::string("../data/") + argv[2];
    std::string pyPredictionsCsv = std::string("../data/") + argv[3];
    std::string cppPredictionsCsv = std::string("../data/") + argv[4];

    // Load the random forest from JSON
    RandomForest forest;
    forest.loadFromJson(forestJson);

    // Read the test data
    std::vector<std::vector<double>> testData;
    readCSV(testCsv, testData);

    // Containers for vote statistics and final predictions
    std::vector<std::vector<int>> voteStats;
    std::vector<std::string> finalPredictions;

    // Process each sample
    for (const auto& sample : testData) {
        // forest.predict should return a pair: vector of vote counts and the predicted class label.
        auto [votes, prediction] = forest.predict(sample);
        voteStats.push_back(votes);
        finalPredictions.push_back(prediction);
    }

    // Read Python predictions from the provided CSV
    std::vector<std::string> py_predictions;
    readPredictionsCSV(pyPredictionsCsv, py_predictions);

    // Compare predictions between Python and C++
    comparePredictions(py_predictions, finalPredictions);

    // Save results to CSV file (votes and final predictions)
    saveCSV(cppPredictionsCsv, voteStats, finalPredictions, forest.classLabels);

    std::cout << "Test completed, results saved in 'iris_test_votes.csv'" << std::endl;

    return 0;
}
