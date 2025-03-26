
import numpy as np
import json
import urllib.parse
import matplotlib.pyplot as plt

def get_visualization_url(seg_id:int, error_features:list[list[float]], em_data_url:str, segmentation_url:str, local_host: bool = True, voxel_resolution:np.ndarray = np.array([16,16,40]), crossSectionScale:float = 0.5, projectionScale:float = 1500, l2_nodes= None)->list[str]:
    """
        This function returns the list of urls for a given neuron. It will look at the selected nodes and give back an url for each selected node.
        Careful: the JSON state is hardcoded, if you want to change it, you need to change it here.
    """
    if error_features.ndim == 1:
        print("No Error Features associated with this segment. Position will be put to 0,0,0")
        merge_coords = []
        split_coords = []
        voxel_coordinate = [0,0,0]
    else:
        error_nodes_coords = error_features[:, :3]
        error_nodes_types = error_features[:, 3]
        merge_coords = error_nodes_coords[error_nodes_types == 0]
        split_coords = error_nodes_coords[error_nodes_types == 1]
        node_number = 0 # TODO: change this to the node number you want to visualize, takes you to the position of the node chosen.
        voxel_coordinate = [
            error_nodes_coords[node_number][0].item() / voxel_resolution[0],
            error_nodes_coords[node_number][1].item() / voxel_resolution[1],
            error_nodes_coords[node_number][2].item() / voxel_resolution[2],
        ]
    merge_annotations = []
    split_annotations = []
    for i, coord in enumerate(merge_coords):
        merge_annotations.append({
            "point": coord.tolist(),
            "type": "point",
            "id": f"annotation_{i}"
        })
    for i, coord in enumerate(split_coords):
        split_annotations.append({
            "point": coord.tolist(),
            "type": "point",
            "id": f"annotation_{i}"
        })
    base_url = "http://localhost:8000/client/#!" if local_host else "https://neuroglancer-demo.appspot.com/#!"


    if l2_nodes is not None:
        segments = list(map(str, l2_nodes.tolist()))
    else:
        segments =[]
    json_state = {
            "dimensions": {
                        "x": [
                        voxel_resolution[0]*1e-9,
                        "m"
                        ],
                        "y": [
                        voxel_resolution[1]*1e-9,
                        "m"
                        ],
                        "z": [
                        voxel_resolution[2]*1e-9,
                        "m"
                        ]
                    },
            "layers": [
                {
            "source": em_data_url,
            "type": "image",
            "tab": "source",
            "name": "EM-image"
        },
        {
            "tab": "segments",
            "source": segmentation_url,
            "type": "segmentation",
            "segments": [str(seg_id)],
            "color": "#0000FF",
            "colorSeed": 883605311,
            "name": "Seg ID"
        },
        {
            "tab": "segments",
            "source": segmentation_url,
            "type": "segmentation",
            "segments": segments,
            "color": "#FF0000",
            "colorSeed": 883605311,
            "name": "L2 Segments"
        },
        {
            "tool": "annotatePoint",
            "type": "annotation",
            "transform": {
            "outputDimensions": {
                "x": [
                1.8e-8,
                "m"
                ],
                "y": [
                1.8e-8,
                "m"
                ],
                "z": [
                4.5e-8,
                "m"
                ]
            },
            "inputDimensions": {
                "0": [
                voxel_resolution[0]*1e-9,
                "m"
                ],
                "1": [
                voxel_resolution[1]*1e-9,
                "m"
                ],
                "2": [
                voxel_resolution[2]*1e-9,
                "m"
                ]
                }
            },
            "annotations": merge_annotations,
            "tab": "annotations",
            "name": "Merge",
            "annotationColor": "#00FF00"
        },
        {
            "tool": "annotatePoint",
            "type": "annotation",
            "transform": {
            "outputDimensions": {
                "x": [
                1.8e-8,
                "m"
                ],
                "y": [
                1.8e-8,
                "m"
                ],
                "z": [
                4.5e-8,
                "m"
                ]
            },
            "inputDimensions": {
                "0": [
                voxel_resolution[0]*1e-9,
                "m"
                ],
                "1": [
                voxel_resolution[1]*1e-9,
                "m"
                ],
                "2": [
                voxel_resolution[2]*1e-9,
                "m"
                ]
            }
            },
            "annotations": split_annotations,
            "tab": "annotations",
            "name": "Split",
            "annotationColor": "#FF0000"
        }       
    ],
    "position": voxel_coordinate,
    "showDefaultAnnotations": False,
    "perspectiveOrientation": [
        0,
        0,
        0,
        1
    ],
    "projectionScale": projectionScale,
    "crossSectionScale": crossSectionScale,
    "jsonStateServer": "https://globalv1.flywire-daf.com/nglstate/post",
    "selectedLayer": {
        "layer": "annotation",
        "visible": True
    },
    "layout": "xy-3d"
    }
    json_string = json.dumps(json_state)
    encoded_string = urllib.parse.quote(json_string)
    return f"{base_url}{encoded_string}"


def visualize_graph_3d(node_features, edge_index, error_features):
    """
    Visualizes a graph in 3D space using the node features and edge index.
    
    Args:
        node_features (torch.Tensor): A tensor of shape (num_nodes, 3) containing node features (x, y, z).
        edge_index (torch.Tensor): A tensor of shape (2, num_edges) containing pairs of node indices that form edges.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates from node features
    x = node_features[:, 0].numpy()
    y = node_features[:, 1].numpy()
    z = node_features[:, 2].numpy()
    
    ax.scatter(x, y, z, c='b', marker='o', s=50, label='Nodes')
    ax.scatter(error_features[:, 0], error_features[:, 1], error_features[:, 2], c='r', marker='x', s=50, label='Error Nodes')
    # Plot the edges
    for i in range(edge_index.size(1)):
        start_node = edge_index[0, i].item()
        end_node = edge_index[1, i].item()
        
        x_coords = [x[start_node], x[end_node]]
        y_coords = [y[start_node], y[end_node]]
        z_coords = [z[start_node], z[end_node]]
        
        ax.plot(x_coords, y_coords, z_coords, c='k',alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Graph Visualization')
    
    plt.legend()
    plt.show()