#pragma once

#include <memory>
#include <vector>

class Node {
public:
    int feature;
    double threshold;
    int value;
    bool isLeaf;
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;

    Node();
};
