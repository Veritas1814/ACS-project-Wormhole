#pragma once

#include <vector>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class DecisionTreeOp3 {
public:
	void loadFromJson(const std::string& filename);
	int buildTree(const json& treeData, int index);
	std::string predict(const std::vector<double>& sample);

private:
	std::vector<int> features;
	std::vector<double> thresholds;
	std::vector<int> values;
	std::vector<bool> isLeaf;
	std::vector<int> leftIndices;
	std::vector<int> rightIndices;
	std::vector<std::string> classLabels;
};
