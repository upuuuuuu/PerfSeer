import networkx as nx
import pickle
import types

from features_define import Feature, Args, MemoryInfo

# NOTE.
# The prefix of the file names for the graph and its corresponding label are the same.
# Execution time refers to the time it takes to process one sample,
# To obtain the time for a whole iteration, you can multiply it by the batch size.

"""
    1. graph
"""
# load file
cg_path = "test.pkl"
with open(cg_path, "rb") as f:
    compute_graph = nx.DiGraph(pickle.load(f))


# 1.1 parse the node features of the graph, based on node features,
# you can extract global features as well as edge features.
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

    # node FLOPs
    node_feature.flops

    # node memory information
    mem_info: MemoryInfo = types.SimpleNamespace(**node_feature.memory_info)
    batch_size = mem_info.batch_size
    weight_size = mem_info.weight_size

    # node arith intensity
    arith_intensity = node_feature.arith_intensity


# 1.2 parse the node features of the graph, using edge information,
# you can derive the connectivity between all nodes in the entire graph.
edges = compute_graph.edges()

"""
    2. label
"""
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
