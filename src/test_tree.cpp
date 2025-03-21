#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "../include/decision_tree.h"

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
	tree.loadFromJson("../data/tree.json");

	std::vector<std::vector<double>> testData;
	readCSV("../data/iris_test.csv", testData);

	std::vector<std::string> predictions;
	for (const auto& sample : testData) {
		predictions.push_back(tree.predict(sample));
	}

	saveCSV("../data/iris_test_result.csv", predictions);

	std::cout << "Predictions saved to iris_test_result.csv" << std::endl;
	return 0;
}