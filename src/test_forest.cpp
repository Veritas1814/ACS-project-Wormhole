#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <map>
#include "../include/decision_forest.h"

using json = nlohmann::json;

// Reads CSV data (skipping header) and converts each row into a vector of doubles.
void readCSV(const std::string& filename, std::vector<std::vector<double>>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    std::string line;
    // Skip header line.
    if (!std::getline(file, line)) {
        std::cerr << "Error: CSV file is empty " << filename << std::endl;
        return;
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string value;
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::exception &e) {
                std::cerr << "Error converting value '" << value << "': " << e.what() << std::endl;
                row.push_back(0.0);
            }
        }
        data.push_back(row);
    }
}

// Saves vote statistics and final predictions to a CSV file.
// First row: class labels, then a "prediction" column header.
// Each subsequent row contains vote counts (one per class) and the final prediction.
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

int main() {
    try {
        // Load the random forest from JSON.
        RandomForest forest;
        forest.loadFromJson("../data/forest.json");

        // Read test samples from CSV.
        std::vector<std::vector<double>> testData;
        readCSV("../data/iris_test.csv", testData);
        if (testData.empty()) {
            std::cerr << "Error: No test data loaded." << std::endl;
            return 1;
        }

        // Containers to store vote statistics and final predictions.
        std::vector<std::vector<int>> voteStats;
        std::vector<std::string> finalPredictions;

        // Process each sample.
        for (const auto& sample : testData) {
            // forest.predict should return a pair: vector of vote counts and the winning class label.
            auto [votes, prediction] = forest.predict(sample);
            voteStats.push_back(votes);
            finalPredictions.push_back(prediction);
        }

        // Save results to a CSV file.
        saveCSV("../data/iris_test_votes.csv", voteStats, finalPredictions, forest.classLabels);
        std::cout << "Test completed, results saved in 'iris_test_votes.csv'" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Exception in main: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}