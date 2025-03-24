#pragma once

#include <vector>
#include <string>
#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class NodeOp2 {
public:
	int feature;
	float threshold;
	int value;
	bool isLeaf;
	std::shared_ptr<NodeOp2> left;
	std::shared_ptr<NodeOp2> right;

	NodeOp2();
};

class DecisionTreeOp2 {
public:
	std::shared_ptr<NodeOp2> root;
	std::vector<std::string> classLabels;

	void loadFromJson(const std::string& filename);
	std::shared_ptr<NodeOp2> buildTree(const json& treeData, int index);
	std::string predict(const std::vector<float>& sample);
};
