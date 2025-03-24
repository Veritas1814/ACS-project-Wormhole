#pragma once

#include <vector>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct NodeOp1 {
	int feature;
	double threshold;
	int value;
	bool isLeaf;
	int leftIndex;
	int rightIndex;
};

class DecisionTreeOp1 {
public:
	std::vector<NodeOp1> nodes;
	std::vector<std::string> classLabels;

	void loadFromJson(const std::string& filename);
	int buildTree(const json& treeData, int index);
	std::string predict(const std::vector<double>& sample);
};
