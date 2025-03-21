#include "decision_forest.h"
#include <iostream>
#include <vector>

int main() {
    RandomForest forest;
    forest.loadFromJson("../data/forest.json");

    std::vector<std::vector<double>> testData = {{5.1, 3.5, 1.4, 0.2}, {6.3, 3.3, 6.0, 2.5}};
    for (const auto& sample : testData) {
        auto [votes, finalPrediction] = forest.predict(sample);
        std::cout << "Predicted: " << finalPrediction << "\n";
    }
    return 0;
}