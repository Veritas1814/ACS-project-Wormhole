#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <map>
#include <unordered_map>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Клас вузла дерева
class Node {
public:
    int feature;        // Індекс ознаки для розділення
    double threshold;   // Поріг для розділення
    int value;          // Значення в листку (клас)
    bool isLeaf;        // Чи є вузол листком
    std::shared_ptr<Node> left;  // Лівий нащадок
    std::shared_ptr<Node> right; // Правий нащадок

    Node() : feature(-1), threshold(0.0), value(-1), isLeaf(false), left(nullptr), right(nullptr) {}
};

// Клас дерева
class DecisionTree {
public:
    std::shared_ptr<Node> root;
    std::map<int, std::string> classLabels;  // Мапа для зберігання міток класів

    // Завантаження дерева з JSON
    void loadFromJson(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Помилка: не вдалося відкрити файл " << filename << std::endl;
            return;
        }

        json treeData;
        try {
            file >> treeData;
        } catch (const std::exception& e) {
            std::cerr << "Помилка при парсингу JSON: " << e.what() << std::endl;
            return;
        }

        // Отримуємо об'єкт "tree"
        auto tree = treeData["tree"];

        // Завантаження міток класів
        for (size_t i = 0; i < tree["classes"].size(); ++i) {
            classLabels[i] = tree["classes"][i];
        }

        // Побудова дерева
        root = buildTree(tree, 0);
    }

    // Рекурсивна функція для побудови дерева
    std::shared_ptr<Node> buildTree(const json& treeData, int index) {
        auto node = std::make_shared<Node>();

        if (treeData["children_left"][index] == -1) {  // Листок
            node->isLeaf = true;

            // Знаходимо індекс максимального значення в масиві
            const auto& values = treeData["value"][index][0];
            int maxIndex = 0;
            double maxValue = values[0];
            for (size_t i = 1; i < values.size(); ++i) {
                if (values[i] > maxValue) {
                    maxValue = values[i];
                    maxIndex = i;
                }
            }
            node->value = maxIndex;  // Вибираємо клас з найбільшою ймовірністю
        } else {  // Внутрішній вузол
            node->feature = treeData["feature"][index];
            node->threshold = treeData["threshold"][index];
            node->left = buildTree(treeData, treeData["children_left"][index]);
            node->right = buildTree(treeData, treeData["children_right"][index]);
        }

        return node;
    }

    // Функція для передбачення
    std::string predict(const std::vector<double>& sample) {
        auto node = root;
        while (!node->isLeaf) {
            if (sample[node->feature] < node->threshold) {
                node = node->left;
            } else {
                node = node->right;
            }
        }
        // Ensure correct mapping of the predicted class
        return classLabels[node->value];
    }


    // Функція для читання CSV файлу
    void CSVReader(const std::string& filename, std::vector<std::vector<double>>& data, std::vector<int>& labels) {
        std::ifstream file(filename);
        std::string line;
        std::unordered_map<std::string, int> classMapping;
        int classIndex = 0;

        std::getline(file, line);  // Пропускаємо перший рядок (заголовок)

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::vector<double> row;
            std::string value;
            while (std::getline(ss, value, ',')) {
                try {
                    row.push_back(std::stod(value));  // Додаємо значення ознаки
                } catch (...) {
                    if (classMapping.find(value) == classMapping.end()) {
                        classMapping[value] = classIndex++;
                    }
                    labels.push_back(classMapping[value]);  // Додаємо мітку класу
                }
            }
            row.pop_back();  // Видаляємо останню колонку, яка є міткою
            data.push_back(row);  // Додаємо дані в масив
        }
    }
};

int main() {
    DecisionTree tree;
    tree.loadFromJson("decision_tree.json");

    std::vector<std::vector<double>> testData;
    std::vector<int> testLabels;

    // Завантажуємо тестові дані з CSV файлу
    tree.CSVReader("iris_test.csv", testData, testLabels);

    // Передбачення
    for (const auto& sample : testData) {
        std::string predictedClass = tree.predict(sample);
        std::cout << "Predicted Class: " << predictedClass << "\n";
    }

    return 0;
}
