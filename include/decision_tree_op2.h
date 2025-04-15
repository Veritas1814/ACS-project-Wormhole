#pragma once

#include <vector>
#include <string>
#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class DecisionTreeOp2 {
private:
	struct NodeOp2 {
		int feature;
		float threshold;
		int value;
		bool isLeaf;
		std::shared_ptr<NodeOp2> left;
		std::shared_ptr<NodeOp2> right;
	};

    std::shared_ptr<NodeOp2> root;
	std::vector<std::string> classLabels;

public:
	void loadFromJson(const std::string& filename);
	std::shared_ptr<NodeOp2> buildTree(const json& treeData, int index);
	int predict(const std::vector<float>& sample) noexcept;

};
