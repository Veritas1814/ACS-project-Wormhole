#pragma once

#include <vector>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;


class DecisionTreeOp4 {
public:

	void loadFromJson(const std::string& filename);
	void buildTree(const json& treeData);
	std::string predict(const std::vector<double>& sample);

private:
	struct NodeOp4 {
		int feature;
		double threshold;
		int value;
		bool isLeaf;
		std::shared_ptr<NodeOp4> leftIndex;
		std::shared_ptr<NodeOp4> rightIndex;
	};

	std::vector<std::shared_ptr<NodeOp4>> nodes;
	std::vector<std::string> classLabels;
};
