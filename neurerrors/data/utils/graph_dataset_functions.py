from logging import root
import torch
import os
from tqdm import tqdm
from torch_geometric.data import Data
import caveclient
import numpy as np
import pandas as pd

def build_pyg_graph(
    client,
    seg_id, 
    attributes_list, 
    verbose=False, 
    save_path=None
):
    """
    Changing actual features of the nodes is NOW supported. 
    """
    error_flag = False
    # Step 1: Fetch level 2 nodes from the client
    lvl2nodes = client.chunkedgraph.get_leaves(seg_id, stop_layer=2)
    # Step 2: Fetch attributes for the nodes
    l2stats = client.l2cache.get_l2data(lvl2nodes, attributes=attributes_list)
    l2df = pd.DataFrame(l2stats).T
    l2df['node_id'] = l2df.index  # Retain node IDs
    if verbose:
        print("L2 dataframe acquired, of shape: ", l2df.shape)
    # Step 3: Fill NaN values with 0
    l2df = l2df.infer_objects(copy=False)
    l2filled = l2df.fillna(0)
    # Step 4: Extract individual features dynamically based on attributes_list
    features = []
    problematic_indices = []
    if 'rep_coord_nm' in attributes_list:
        # Safely handle 'rep_coord_nm' as a list or array
        rep_coords = []
        for idx, coord in enumerate(l2filled['rep_coord_nm']):
            if isinstance(coord, (list, np.ndarray)) and len(coord) == 3:  # Check if it's a 3D coordinate
                rep_coords.append(coord)
            else:
                if verbose: 
                    print("REP_COORD_NM NOT VALID, error flagged")
                    print("Coord is: ", coord, "root id is ", seg_id)
                error_flag = True
                rep_coords.append([0, 0, 0])  # Default to [0, 0, 0] if not valid
                problematic_indices.append(idx)
        rep_coords = np.array(rep_coords)
        features.append(rep_coords)
    if 'size_nm3' in attributes_list:
        features.append(l2filled['size_nm3'].to_numpy())
    if 'area_nm2' in attributes_list:
        features.append(l2filled['area_nm2'].to_numpy())
    if 'max_dt_nm' in attributes_list:
        features.append(l2filled['max_dt_nm'].to_numpy())
    if 'mean_dt_nm' in attributes_list:
        features.append(l2filled['mean_dt_nm'].to_numpy())
    if 'pca_val' in attributes_list:
        pca_vals = []
        for pca in l2filled['pca_val']:
            if pca == 0:
                pca_vals.append([0, 0, 0])
            elif isinstance(pca_vals, (list, np.ndarray)) and len(pca) == 3:
                pca_vals.append(pca)
            else:
                pca_vals.append([0, 0, 0])
        pca_vals = np.array(pca_vals)
        features.append(pca_vals)
    # Map node IDs to contiguous indices
    original_node_ids = l2filled['node_id'].to_numpy()
    node_id_mapping = {int(node_id): idx for idx, node_id in enumerate(np.unique(original_node_ids))}
  
    # Step 5: Combine the features dynamically
    if features:
        combined_array = np.column_stack(features)
    else:
        print("NO FEATURES")
        combined_array = np.empty((len(l2df), 0))  # Empty if no features are selected

    chunk_size = 2500  # Or any suitable chunk size depending on memory.
    node_features_tensor = torch.empty((len(combined_array), combined_array.shape[1]), dtype=torch.float32)

    # Fill the pre-allocated tensor in chunks
    for start in range(0, len(combined_array), chunk_size):
        end = min(start + chunk_size, len(combined_array))
        # Copy data chunk directly into the pre-allocated tensor
        node_features_tensor[start:end] = torch.tensor(combined_array[start:end], dtype=torch.float32)

    # Step 6: Build edge index using node ID mapping
    edges_client = client.chunkedgraph.level2_chunk_graph(seg_id)
    edges = np.array([[node_id_mapping[src], node_id_mapping[dst]] for src, dst in edges_client])

    if edges.size != 0:
        """ Edges are processed in chunks to avoid memory issues. Since the graph is undirected, we need to double the amount of edges. PyG technically supports these, but it's not implemented."""
        batch_size = 5000  # Adjust this size based on your memory constraints
        num_edges = len(edges)
        edge_index_tensor = []
        for i in range(0, num_edges, batch_size):
            batch_edges = edges[i:i + batch_size]
            batch_tensor = torch.tensor(batch_edges, dtype=torch.long).T  # Transpose each batch
            edge_index_tensor.append(batch_tensor)
        edge_index_tensor = torch.cat(edge_index_tensor, dim=1)
        chunk_size = 2500  # Adjust as needed based on system memory
        total_edges = edge_index_tensor.size(1)
        edge_chunks = []
        for start in range(0, total_edges, chunk_size):
            end = min(start + chunk_size, total_edges)
            edge_chunks.append(edge_index_tensor[:, start:end])
            reverse_chunk = edge_index_tensor[:, start:end].flip(0)
            edge_chunks.append(reverse_chunk)
        edge_index_tensor = torch.cat(edge_chunks, dim=1)
        if verbose:
            print("Final concatenated edge index tensor shape:", edge_index_tensor.shape)
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        print("No edges found, empty edge index tensor created")
    # Step 7: Create PyG Data object
    data = Data(x=node_features_tensor, edge_index=edge_index_tensor)

    inverse_mapping = torch.full((len(node_id_mapping),), -1, dtype=torch.long)
    for node_id, idx in node_id_mapping.items():
        inverse_mapping[idx] = node_id

    data.l2_nodes = inverse_mapping # This allows to check directly node to L2 ID. Which is very useful for remapping to errors.

    # Possible to uncomment. Shows problematic nodes.
    # problematic_nodes = [inverse_mapping[node_id] for node_id in problematic_indices]
    # print("Problematic nodes: ", problematic_nodes)
    # Step 9: Save to file if required
    if save_path != None:
        torch.save(data, save_path)
        if verbose:
            print(f"Data saved to {save_path}")
    return data, error_flag


def create_dataset(
        client, 
        seg_ids, 
        attributes_list, 
        save_path:str=None, 
        intermediate_save:int=5000, 
        show_progress:bool=True, 
        verbose=False
        ):
    """
        Entry function to create a list of PyG data points, consisting of the neurons at the L2 levels. Calling for each seg_id "build_pyg_graph".
        Args are:
            - client: ChunkedGraph base client initialized for the dataset.
            - seg_ids: List of seg_ids as ints. These seg_ids represent the neurons we want.
            - attributes_list: List of items we want each node to have as features. Order doesn't matter. Example: ['rep_coord_nm', 'size_nm3', 'area_nm2', 'max_dt_nm', 'mean_dt_nm', 'pca_val']
            - save_path: String indicating save_path
            - intermediate_save: Int indicating how frequently we want dataset saving. Useful in case of crashing of client (can happen).
            - verbose: Bool to have more intermediate outputs.
        Outputs:
            - dataset: List of PyG data points.
    """
    dataset = []
    iterator = tqdm(enumerate(seg_ids), total=len(seg_ids), desc="Building L2 graphs", unit="ID") if show_progress else enumerate(seg_ids)
    for i, seg_id in iterator:
        try:
            data, error_flag = build_pyg_graph(client, seg_id, attributes_list, verbose=verbose)
            if verbose:
                if error_flag:
                    print("Errors found during build, graph will contain errors or missing values", flush=True)
                else:
                    print(f"Succesful build for segment ID: {seg_id}")
        except Exception as e:
            print(f"Error generating graphs for {seg_id}: {e}", flush=True)
            continue

        metadata = {'seg_id': seg_id}
        data.metadata = metadata
        dataset.append(data)
        if save_path is not None and intermediate_save is not None and i % intermediate_save == 0:
            intermediate_path = save_path.replace('.pt', f'_intermediate_{i}.pt')
            torch.save(dataset, intermediate_path)
            print(f"Dataset saved to {intermediate_path}", flush=True)

    return dataset
