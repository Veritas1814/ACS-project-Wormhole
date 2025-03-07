#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <limits>
#include <unordered_map>

class Node {
public:
    std::shared_ptr<Node> left, right;
    std::vector<std::vector<double>> data;  // Dataset at the node
    std::vector<int> labels;                // Labels for classification
    int bestFeature;                        // Feature selected for split
    double bestThreshold;                   // Threshold for the split
    bool isLeaf;                            // Flag to indicate if the node is a leaf

    Node() : left(nullptr), right(nullptr), bestFeature(-1), bestThreshold(0.0), isLeaf(false) {}

    // Constructor with data and labels
    Node(const std::vector<std::vector<double>>& data, const std::vector<int>& labels)
        : data(data), labels(labels), left(nullptr), right(nullptr), bestFeature(-1), bestThreshold(0.0), isLeaf(false) {}

    // Method to calculate Gini Impurity
    double giniImpurity(const std::vector<int>& labels) {
        std::map<int, int> classCounts;
        for (int label : labels) {
            classCounts[label]++;
        }

        double impurity = 1.0;
        for (const auto& [_, count] : classCounts) {
            double prob = static_cast<double>(count) / labels.size();
            impurity -= prob * prob;
        }
        return impurity;
    }

    // Method to find the best split feature and threshold
    std::pair<int, double> bestSplit() {
        int bestFeature = -1;
        double bestThreshold = 0.0;
        double bestImpurity = std::numeric_limits<double>::max();

        int numFeatures = data[0].size();
        for (int feature = 0; feature < numFeatures; ++feature) {
            std::set<double> uniqueValues;
            for (const auto& row : data) {
                uniqueValues.insert(row[feature]);
            }
            for (double threshold : uniqueValues) {
                std::vector<int> leftLabels, rightLabels;
                for (size_t i = 0; i < data.size(); ++i) {
                    if (data[i][feature] < threshold) {
                        leftLabels.push_back(labels[i]);
                    } else {
                        rightLabels.push_back(labels[i]);
                    }
                }
                double impurity = (leftLabels.size() * giniImpurity(leftLabels) +
                                   rightLabels.size() * giniImpurity(rightLabels)) / labels.size();
                if (impurity < bestImpurity) {
                    bestImpurity = impurity;
                    bestFeature = feature;
                    bestThreshold = threshold;
                }
            }
        }
        return {bestFeature, bestThreshold};
    }

    // Method to split the data based on the best split
    void splitNode() {
        auto [bestFeature, bestThreshold] = bestSplit();
        this->bestFeature = bestFeature;
        this->bestThreshold = bestThreshold;

        std::vector<std::vector<double>> leftData, rightData;
        std::vector<int> leftLabels, rightLabels;

        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i][bestFeature] < bestThreshold) {
                leftData.push_back(data[i]);
                leftLabels.push_back(labels[i]);
            } else {
                rightData.push_back(data[i]);
                rightLabels.push_back(labels[i]);
            }
        }

        if (leftData.size() > 0 && rightData.size() > 0) {
            left = std::make_shared<Node>(leftData, leftLabels);
            right = std::make_shared<Node>(rightData, rightLabels);
        }
    }
};

// Tree class to represent the decision tree
class Tree {
public:
    std::shared_ptr<Node> root;
    std::unordered_map<int, std::string> classLabels;  // Mapping from index to class label

    Tree() : root(nullptr) {}

    // Destructor is automatically handled by shared_ptr


    // Method to fit the decision tree to the dataset
    void fit(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int depth = 0, int maxDepth = 27) {
        root = std::make_shared<Node>(data, labels);
        buildTree(root, labels, depth, maxDepth);
    }

    // Method to recursively build the tree
    void buildTree(std::shared_ptr<Node> node, const std::vector<int>& labels, int depth, int maxDepth) {
        // Base case: If no more splitting is possible or max depth is reached
        if (depth >= maxDepth || node->giniImpurity(node->labels) < 0.01 || node->labels.size() <= 1) {
            node->isLeaf = true;
            return;
        }

        node->splitNode();
        buildTree(node->left, labels, depth + 1, maxDepth);
        buildTree(node->right, labels, depth + 1, maxDepth);
    }

    // Method to traverse the tree and print the nodes
    void traverse(std::shared_ptr<Node> node) {
        if (!node) return;
        std::cout << "Feature: " << node->bestFeature << " Threshold: " << node->bestThreshold << "\n";
        if (node->left) traverse(node->left);
        if (node->right) traverse(node->right);
    }

    // Method to make predictions based on a sample
    std::string predict(const std::vector<double>& sample) {
        std::shared_ptr<Node> node = root;
        while (!node->isLeaf) {
            if (sample[node->bestFeature] < node->bestThreshold)
                node = node->left;
            else
                node = node->right;
        }
        return classLabels[node->labels[0]];  // Output the class label instead of index
    }

    // Method to read data from a CSV file
    void CSVReader(const std::string& filename, std::vector<std::vector<double>>& data, std::vector<int>& labels) {
        std::ifstream file(filename);
        std::string line;
        std::unordered_map<std::string, int> classMapping;
        int classIndex = 0;

        // Skip header line
        std::getline(file, line);

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::vector<double> row;
            std::string value;
            while (std::getline(ss, value, ',')) {
                try {
                    row.push_back(std::stod(value));
                } catch (...) {
                    // Handle class label
                    if (classMapping.find(value) == classMapping.end()) {
                        classMapping[value] = classIndex++;
                        classLabels[classIndex - 1] = value;  // Store the label
                    }
                    labels.push_back(classMapping[value]);
                }
            }
            row.pop_back();  // Remove the last incorrect element if it exists
            data.push_back(row);
        }
    }
};

// Main function to demonstrate training and testing
int main() {
    // Initialize tree and data structures
    Tree tree;
    std::vector<std::vector<double>> trainData;
    std::vector<int> trainLabels;

    // Load training data from train.csv
    std::string trainFile = "iris_train.csv";
    tree.CSVReader(trainFile, trainData, trainLabels);

    // Train the decision tree
    tree.fit(trainData, trainLabels);
    std::cout << "Training Complete!\n";

    // Load test data from test.csv
    std::vector<std::vector<double>> testData;
    std::vector<int> testLabels; // Optional, if you have true labels for testing
    std::string testFile = "iris_test.csv";
    tree.CSVReader(testFile, testData, testLabels);

    // Classify each instance from the test set
    std::cout << "Classifying Test Instances:\n";
    for (const auto& sample : testData) {
        std::string predictedClass = tree.predict(sample);
        std::cout << "Predicted Class: " << predictedClass << "\n";
    }

    return 0;
}
