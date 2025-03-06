#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>

using namespace std;

class Node {
public:
    bool isLeaf = false;
    string label;
    string splittingAttribute;
    string splittingValue;
    Node* leftChild = nullptr;
    Node* rightChild = nullptr;

    Node() = default;
    Node(const string& label) : isLeaf(true), label(label) {}
};

struct Table {
    vector<string> attrName;
    vector<vector<string>> data;
};

class InputReader {
private:
    ifstream fin;
    Table table;

public:
    InputReader(const string& filename) {
        fin.open(filename);
        if (!fin) {
            cerr << filename << " file could not be opened\n";
            exit(1);
        }
        parse();
    }

    void parse() {
        string line;
        bool isAttrName = true;
        while (getline(fin, line)) {
            vector<string> row;
            stringstream ss(line);
            string cell;
            while (getline(ss, cell, ',')) {
                row.push_back(cell);
            }

            if (isAttrName) {
                table.attrName = row;
                isAttrName = false;
            } else {
                table.data.push_back(row);
            }
        }
    }

    Table getTable() {
        return table;
    }
};

class DecisionTreeClassifier {
public:
    Node* root = nullptr;
    Table table;
    int targetAttrIndex;

    DecisionTreeClassifier(Table table, int targetAttrIndex) : table(table), targetAttrIndex(targetAttrIndex) {
        root = buildTree(table.data);
    }

    double giniImpurity(const vector<vector<string>>& data) {
        map<string, int> labelCount;
        for (const auto& row : data) {
            labelCount[row[targetAttrIndex]]++;
        }
        double impurity = 1.0;
        int total = data.size();
        for (const auto& [label, count] : labelCount) {
            double prob = static_cast<double>(count) / total;
            impurity -= prob * prob;
        }
        return impurity;
    }

    double informationGain(const vector<vector<string>>& left, const vector<vector<string>>& right, double currentUncertainty) {
        double p = static_cast<double>(left.size()) / (left.size() + right.size());
        return currentUncertainty - p * giniImpurity(left) - (1 - p) * giniImpurity(right);
    }

    Node* buildTree(vector<vector<string>> data) {
        if (data.empty()) return nullptr;

        set<string> uniqueLabels;
        for (auto& row : data) {
            uniqueLabels.insert(row[targetAttrIndex]);
        }

        if (uniqueLabels.size() == 1) {
            return new Node(*uniqueLabels.begin());
        }

        double bestGain = 0.0;
        int bestAttr = 0;
        string bestValue;
        double currentUncertainty = giniImpurity(data);

        for (int attr = 0; attr < table.attrName.size() - 1; ++attr) {
            set<string> values;
            for (const auto& row : data) {
                values.insert(row[attr]);
            }
            for (const auto& value : values) {
                vector<vector<string>> left, right;
                for (const auto& row : data) {
                    if (row[attr] == value) {
                        left.push_back(row);
                    } else {
                        right.push_back(row);
                    }
                }
                if (left.empty() || right.empty()) continue;
                double gain = informationGain(left, right, currentUncertainty);
                if (gain >= bestGain) {
                    bestGain = gain;
                    bestAttr = attr;
                    bestValue = value;
                }
            }
        }

        Node* node = new Node();
        node->splittingAttribute = table.attrName[bestAttr];
        node->splittingValue = bestValue;


        vector<vector<string>> leftData, rightData;
        for (auto& row : data) {
            if (row[bestAttr] == bestValue) {
                leftData.push_back(row);
            } else {
                rightData.push_back(row);
            }
        }

        node->leftChild = buildTree(leftData);
        node->rightChild = buildTree(rightData);

        return node;
    }

    string predict(vector<string> row) {
        Node* currentNode = root;
        while (!currentNode->isLeaf) {
            if (row[0] == currentNode->splittingValue) {
                currentNode = currentNode->leftChild;
            } else {
                currentNode = currentNode->rightChild;
            }
        }
        return currentNode->label;
    }
};

void printTree(Node* node, const string& prefix = "", bool isLeft = true) {
    if (node == nullptr) return;

    if (node->isLeaf) {
        cout << prefix << (isLeft ? "├── " : "└── ") << "Label: " << node->label << "\n";
        return;
    }

    cout << prefix << (isLeft ? "├── " : "└── ")
         << node->splittingAttribute << " = " << node->splittingValue << "\n";

    printTree(node->leftChild, prefix + (isLeft ? "│   " : "    "), true);
    printTree(node->rightChild, prefix + (isLeft ? "│   " : "    "), false);
}

int main() {
    string trainFile = "iris_train2.csv";
    string testFile = "output_py.csv";

    InputReader trainReader(trainFile);
    Table trainTable = trainReader.getTable();

    int targetAttrIndex = trainTable.attrName.size() - 1;
    DecisionTreeClassifier classifier(trainTable, targetAttrIndex);

    cout << "\nDecision Tree Structure:\n";
    printTree(classifier.root);

    InputReader testReader(testFile);
    Table testTable = testReader.getTable();

    cout << "\nTest Results:\n";
    for (const auto& row : testTable.data) {
        cout << "Actual: " << row[targetAttrIndex]
             << " | Predicted: " << classifier.predict(row) << endl;
    }

    return 0;
}
