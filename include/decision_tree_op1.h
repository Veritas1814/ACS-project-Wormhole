#pragma once

#include <vector>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class DecisionTreeOp1 {
public:
	void loadFromJson(const std::string& filename);
	int buildTree(const json& treeData, int index);
	int predict(const std::vector<double>& sample) noexcept;

private:
	struct NodeOp1 {
		int feature;
		double threshold;
		int value;
		bool isLeaf;
		int leftIndex;
		int rightIndex;
	};

	std::vector<NodeOp1> nodes;
	std::vector<std::string> classLabels;

};
