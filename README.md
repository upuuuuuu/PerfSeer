# A Generalized and Accurate Deep Learning Models Performance Predictor towards Cluster Schedulers
## Overview
GAPP enables accurate, convenient, and rapid predictions of various performance metrics for various types of DL jobs, addressing the limitations of existing performance predictor in terms of generality and accuracy.  

We open sourced the performance metrics dataset used by GAPP.  
This dataset spans various styles of network architectures, including GoogLeNet, VGG, ResNe(X)t, MobileNet, and DenseNet, covering a wide range of floating point operations (FLOPs) from 49M to 22T. Additionally, we collect the execution time, GPU memory usage, and GPU Streaming Multiprocessor (SM) utilization for these model configurations during both the training and inference phases on the Nvidia GeForce RTX 3090.

## Download the dataset 
Download and unpack our dataset from the [public google drive folder](https://drive.google.com/drive/folders/15anTR-bBTTfvXx9aQXp1BlMcqXjJsmmW?usp=sharing)

## Using the dataset
Example usage(see util_dataset/example.py for a full runnable example)
- parse graph
```python
import networkx as nx
import pickle
import types

from features_define import Feature, Args, MemoryInfo

# load file
cg_path = "test.pkl"
with open(cg_path, "rb") as f:
    compute_graph = nx.DiGraph(pickle.load(f))

# 1.1 parse the node features of the graph
for node_idx in compute_graph.nodes():
    # get node features
    node_feature: Feature = types.SimpleNamespace(
        **compute_graph.nodes[node_idx]["feature"]
    )

    # node type
    type = node_feature.type

    # node args
    args: Args = types.SimpleNamespace(**node_feature.args)
    kernel_size = args.conv_kernel_size

    # node memory information
    mem_info: MemoryInfo = types.SimpleNamespace(**node_feature.memory_info)
    batch_size = mem_info.batch_size
    weight_size = mem_info.weight_size

# 1.2 parse the node features of the graph, using edge information,
# you can derive the connectivity between all nodes in the entire graph.
edges = compute_graph.edges()
```
- parse label
```python
label_path = "test.txt"
with open(label_path, "r") as f:
    label_dict = eval(f.read())
    train_label = label_dict["train"]
    infer_label = label_dict["infer"]
    # get metric label
    (
        time,
        peak_sm_util,
        average_memory_util,
        average_memory_usuage,
        peak_sm_util,
        peak_memory_util,
        peak_memory_usuage,
    ) = train_label.split("|")
```
