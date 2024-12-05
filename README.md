# Evaluate-Graph-Metrics
Evaluate the similarity between two graphs, e.g. ground truth vs predicted skeletons of blood vessel segmentation.

## Networkx package

We make use of the [networkx](https://networkx.org/documentation/stable/reference/introduction.html) package.
By their [naming convention](https://networkx.org/documentation/stable/reference/algorithms/tree.html) we only consider graphs which are _branchings_. A branching is a directed forest with each node having, at most, one parent. So the maximum in-degree is equal to 1. Nodes without parents are called _roots_. Nodes without children are called _leaves_.
