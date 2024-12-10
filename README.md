# Evaluate-Graph-Metrics
Evaluate the similarity between two graphs, e.g. ground truth vs predicted skeletons of blood vessel segmentation.

You can call the code like this:
``` shell
python main.py --gt tests/toygraph_GT.swc --pred tests/toygraph_predicted.swc
```
If you call only `main.py` without specifying the path variables for the ground truth and predicted graph, toy graphs from `project/toydata.py` will be used.

## Networkx package

We make use of the [networkx](https://networkx.org/documentation/stable/reference/introduction.html) package.
By their [naming convention](https://networkx.org/documentation/stable/reference/algorithms/tree.html) we only consider graphs which are _branchings_. A branching is a directed forest with each node having, at most, one parent. So the maximum in-degree is equal to 1. Nodes without parents are called _roots_. Nodes without children are called _leaves_.

## Metrics
Currently the following metrics are available:
* `Betti-0 error`
* `Betti-1 error`
* `Precision`, `Recall` and `F1-score` with respect to false merges (FP) and false splits (FN)
* `Street Mover Distance` (or `Sinkhorn distance`) between two graphs viewed as point clouds 
* `Tree Edit Distance` (tbd)

## Usage
* Create a virtual environment with all necessary packages: torch, numpy, networkx (TODO: Create requirements.txt)
* Run [`main.py`](https://github.com/Kainmueller-Lab/Evaluate-Graph-Metrics/blob/main/main.py) to print all current graph similarity metrics applied on two graphs. You can either use predefined toy graphs or load your own target and source graphs.
