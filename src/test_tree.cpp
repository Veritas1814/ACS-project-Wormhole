#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "decision_tree.h"
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

void saveCSV(const std::string& filename, const std::vector<int>& predictions) {
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

void comparePredictions(const std::vector<int>& py_preds, const std::vector<int>& cpp_preds) {
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
        std::cerr << "Usage: " << argv[0] << " <tree_json> <test_csv> <python_predictions_csv> <cpp_predictions_csv>" << std::endl;
        return 1;
    }

    // Get file paths from arguments
    std::string treeJson = std::string("../data/") + argv[1];
    std::string testCsv = std::string("../data/") + argv[2];
    std::string pyPredictionsCsv = std::string("../data/") + argv[3];
    std::string cppPredictionsCsv = std::string("../data/") + argv[4];

    // Load the decision tree
    DecisionTree tree;
    tree.loadFromJson(treeJson);

    // Read the test data
    std::vector<std::vector<double>> testData;
    readCSV(testCsv, testData);

    // Make predictions
    std::vector<int> cpp_predictions;
    for (const auto& sample : testData) {
        cpp_predictions.push_back(tree.predict(sample)); // Assuming predict returns int
    }

    // Read Python predictions from the provided CSV
    std::vector<int> py_predictions;
    readPredictionsCSV(pyPredictionsCsv, py_predictions);

    // Compare predictions
    comparePredictions(py_predictions, cpp_predictions);

    // Save C++ predictions to CSV
    saveCSV(cppPredictionsCsv, cpp_predictions);

    std::cout << "Predictions saved to " << cppPredictionsCsv << std::endl;

    return 0;
}
