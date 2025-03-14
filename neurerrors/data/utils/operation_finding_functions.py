import torch
import numpy as np
from torch_geometric.data import Data
import caveclient
from tqdm import tqdm
import networkx as nx
import os

def build_graph_ground_truth(data, error_features, verbose=False):
    """
        Builds a new attribute "problematic" in the graph, which is a tensor of shape (num_nodes, 1) containing the ground truth for each node. Associated by distance to the error. 1 if problematic, 0 otherwise.
    """
    coords = data.x.numpy()[:, :3]
    error_features = np.array(error_features)
    # Initialize the problematic attribute with zeros
    data.l2_error_weights = torch.zeros((data.num_nodes, 1), dtype=torch.float)
    if len(error_features) == 0:
        if verbose:
            print("No avg coords to build ground truth", flush=True)
        return data
    error_features_coords = error_features[:, :3]  
    weights = error_features[:, 4]
    for point, weight in zip(error_features_coords, weights):
        # Calculate distances from each node to the current point and check if they are valid (not NaN or Inf)
        distances = np.linalg.norm(coords - point, axis=1)
        if np.any(np.isnan(distances)) or np.any(np.isinf(distances)):
            if verbose:
                print(f"Warning: Invalid distances for point {point}. Skipping...")
            continue
        closest_node_idx = np.argmin(distances)
        data.l2_error_weights[closest_node_idx] += 1.0 * weight
        if verbose:
            print("Weight is: ", weight)
            print("Attributed to node number ", closest_node_idx, "who has coordinates: ", coords[closest_node_idx])
    return data

def weight_operation(client, prev_seg_id, current_seg_id, merge, split):
    """
    Computes the weight of a merge or split operation based on changes in leaf nodes 
    between two segmentation roots in the chunked graph.

    - If `merge` is True, returns the number of newly added leaf nodes.
    - If `split` is True, returns the smaller count between remaining and removed leaves.

    Args:
        client: ChunkedGraph client instance.
        prev_seg_id (int): Root ID before the operation.
        current_seg_id (int): Root ID after the operation.
        merge (bool): Whether the operation is a merge.
        split (bool): Whether the operation is a split.

    Returns:
        int: Weight of the operation based on the number of affected leaf nodes.
    """
    lvl2nodes1 = client.chunkedgraph.get_leaves(prev_seg_id,stop_layer=2)
    lvl2nodes2 = client.chunkedgraph.get_leaves(current_seg_id,stop_layer=2)
    # find the amount of added leaves
    added_leaves = set(lvl2nodes2) - set(lvl2nodes1)
    removed_leaves = set(lvl2nodes1) - set(lvl2nodes2)
    if merge:
        return len(added_leaves)
    if split:
        return min(len(lvl2nodes1) - len(removed_leaves), len(removed_leaves))

def get_descendants_with_operations(client, seg_id, voxel_resolution, verbose=False):
    """
    Retrieves all descendant nodes of a given root in the chunked graph and identifies 
    merge or split operations associated with the root.

    - Extracts the lineage graph and filters operations that modify the given root looking at intersection of l1 leaves.
    - Determines the type (merge or split) of each operation.
    - Computes the average operation coordinates and assigns a weight.
    - Filters out operations with invalid coordinates.

    Args:
        client: ChunkedGraph client instance.
        seg_id (int): Root ID for which to retrieve descendants and operations.
        voxel_resolution (np.ndarray): Scaling factor for operation coordinates.

    Returns:
        tuple: (set of descendant node IDs, set of operation IDs, numpy array of operation details)
            - operation details format: numpy array of shape (num_operations, 6) containing [x, y, z, operation_type, operation_weight, operation_id]
    """
    CG = client.chunkedgraph # for simplicity
    try:
        past = CG.get_root_timestamps([seg_id])
        nxgraph = CG.get_lineage_graph(seg_id, as_nx_graph=True, exclude_links_to_past=True, timestamp_past=past[0])
    except Exception as e:
        print(f"Error getting lineage graph for {seg_id}: {e}")
        return [], set(), []

    seg_id_edges = CG.get_leaves(seg_id)
    descendants = nx.descendants(nxgraph, seg_id)
    operations_set = set()


    operation_ids = nx.get_node_attributes(nxgraph, 'operation_id')
    operation_id_list = [
                            operation_ids[node]
                            for node in descendants
                            if node in operation_ids
                        ]
    operations_to_root = {operation_ids[node]: node for node in descendants if node in operation_ids}
    operation_id_list = list(set(operation_id_list)) #get rid off non duplicate operation ids
    all_op_details = {}
    error_features = []

    for i in range(0, len(operation_id_list), 100):
        if verbose:
            print("Progress of operation details processing: ", (i+1)/len(operation_id_list) * 100)
        chunk = operation_id_list[i:i + 100]
        
        op_details = CG.get_operation_details(chunk)
        all_op_details.update(op_details)

    for i, op in enumerate(operation_id_list):
        if verbose and i % 100 == 0:
            print("Progress of operation details filtering: ", (i+1)/len(operation_id_list) * 100)
        merge = False
        split = False
        op_details = all_op_details[str(op)]
        
        if 'added_edges' in op_details.keys() and len(set(op_details['added_edges'][0]).intersection(seg_id_edges)) != 0:
            merge = True
        elif 'removed_edges' in op_details.keys() and len(set(op_details['removed_edges'][0]).intersection(seg_id_edges)) != 0:
            split = True
        else:
            continue
        source_coords = np.array(op_details['source_coords'])
        sink_coords = np.array(op_details['sink_coords'])
        MSOC = np.mean(source_coords, axis=0) 
        MSIC = np.mean(sink_coords, axis=0)   
        avg_coord = (MSIC + MSOC) / 2
        if avg_coord[0] > 10**12 or avg_coord[1] > 10**12 or avg_coord[2] > 10**12:
            if verbose:
                print("Operation out of bounds, probably a chunkedgraph mistake")
            continue

        operation_type = 0 if merge else 1 if split else 2
        resolution = voxel_resolution
        avg_coord *= resolution
        try:
            prev_seg_id = op_details['roots'][0] # we want to get the previous root id in the lineage graph. The problem with op_details['roots'][0] is that it can give some random root id 
            current_seg_id = operations_to_root[op] # this gets the node from which the operation is coming from
            parents = list(nxgraph.predecessors(current_seg_id))
            if parents:
                prev_seg_id = parents[0]
            else:
                print(f"Node {current_seg_id} has no parents.")
            operation_weight = weight_operation(client, prev_seg_id, current_seg_id, merge, split)
        except Exception as e:
            if verbose:
                print(f"Error weighting operation {op}: {e}")
            operation_weight = 1
        error_features.append([*avg_coord, operation_type, operation_weight, op])
        operations_set.add(op)
    if verbose:
        print("Amount of operations that were added for seg_id: ", seg_id, "is: ", len(operations_set))
    #Convert error_features to numpy array
    error_features = np.array(error_features)
    return descendants, operations_set, error_features


def build_error_features(client, data:Data=None, seg_id:int=None, voxel_resolution:np.ndarray=np.array([16,16,40]), verbose=False):
    """
    Builds a dataset of the error coordinates for each segment ID by making a forward pass in the graph of operations.
    This can be used as a stand-alone function to build a dataset of error coordinates for a given segment ID.
    If matched with a data point it will associate the error features to the graph nodes that are closest and build the .l2_error_weights attribute.
    """
    _, __, error_features = get_descendants_with_operations(client, seg_id, voxel_resolution, verbose=verbose)
    if data is not None:
        data = build_graph_ground_truth(data, error_features, verbose=verbose)
        return data, error_features
    else:
        return error_features

def create_error_dataset_from_graph_list(
        client:caveclient.CAVEclient, 
        data_list:list[Data], 
        voxel_resolution:np.ndarray=np.array([16,16,40]),
        save_path:str=None, intermediate_save:int=5000, 
        show_progress:bool=True, 
        verbose=False
        ):
    """
        Processes each graph in the list, finds errors in the chunkedgraph operations associated to the seg_id.
        Builds a new attribute "error_features" in the graph, which is a tensor of shape (num_nodes, 5) containing the error features for each node.
        The 6 features are:
            - x,y,z coordinates of the error in nm
            - operation type (0 for merge, 1 for split, 2 for other)
            - operation weight
            - operation id
        Builds a new attribute "l2_error_weights" in the graph, which is a tensor of shape (num_nodes, 1) containing the ground truth for each node. Associated by distance to the error. 1>= if problematic, 0 otherwise.
    """
    iterator = tqdm(enumerate(data_list), total=len(data_list), desc="Finding errors and associating them to graphs", unit="graph") if show_progress else enumerate(data_list)
    for i, data in iterator:
        if show_progress:
            iterator.set_postfix(seg_id=data.metadata['seg_id'])
        data, error_features = build_error_features(client, data, data.metadata['seg_id'], voxel_resolution, verbose=verbose)
        data.error_features = torch.tensor(error_features, dtype=torch.float32)
        
        if save_path is not None and intermediate_save is not None and i % intermediate_save == 0:
            intermediate_path = save_path.replace('.pt', f'_edited_{i}.pt')
            torch.save(data_list, intermediate_path)
            print(f"Dataset saved to {intermediate_path}", flush=True)

    if save_path is not None:
        torch.save(data_list, save_path)
        print(f"Dataset saved to {save_path}", flush=True)
    return data_list