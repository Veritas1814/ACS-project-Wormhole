#pragma once

#include <vector>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct NodeOp4 {
	int feature;
	double threshold;
	int value;
	bool isLeaf;
	int leftIndex;
	int rightIndex;
};

class DecisionTreeOp4 {
public:
	std::vector<NodeOp4> nodes;
	std::vector<std::string> classLabels;

	void loadFromJson(const std::string& filename);
	void buildTree(const json& treeData);
	std::string predict(const std::vector<double>& sample);
};
