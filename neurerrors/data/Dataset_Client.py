from neurerrors.data.utils.graph_dataset_functions import create_dataset
from neurerrors.data.utils.operation_finding_functions import create_error_dataset_from_graph_list, build_error_features
from neurerrors.data.utils.visualization_functions import get_visualization_url, visualize_graph_3d
from neurerrors.models.training.model_architecture.Neuron_GNN import GCNGlobalPredictor3_5
from neurerrors.models.training.evaluate import evaluate_model
import torch
from tqdm import tqdm
from torch_geometric.data import Data
import numpy as np
import caveclient
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader


""" 
    ##############################################################################################################################
    Example of how to use the Dataset_Client class:

    datastack_name = "flywire_fafb_public"
    client = caveclient.CAVEclient(datastack_name)
    client.materialize.version = 783

    input_path = 'src/neurerrors/dataset_acquisition/seg_ids/ID_tests.txt'
    attributes_list = ['rep_coord_nm', 'size_nm3', 'area_nm2', 'max_dt_nm', 'mean_dt_nm', 'pca_val']

    dataset_client = Dataset_Client(client)
    old_seg_ids = dataset_client.get_old_seg_ids_from_txt_file(input_path)

    dataset = dataset_client.build_graph_dataset(old_seg_ids, attributes_list)
    print(dataset)

    dataset_with_error_features = dataset_client.associate_error_to_graph_dataset(dataset)
    
    # Show in Neuroglancer
    url = dataset_client.get_url_for_visualization(seg_id=dataset[0].metadata['seg_id'], error_features=dataset[0].error_features, local_host=True)
    print(url)

    ##############################################################################################################################
"""


class Dataset_Client():
    """
    A class to build a graph dataset of past IDs from a list of root IDs. Allows extension to datasets contained in the caveclient chunkedgraph.
    Can acquire independently from the graph dataset a dataset of the error coordinates for each segment ID by making a forward pass in the graph of operations.
    """
    def __init__(self, client: caveclient.CAVEclient = None):
        self.client = client
        if self.client:
            self.CG = self.client.chunkedgraph
        else:
            self.CG = None 

    def verify_size(
        self, 
        seg_id: int, 
        treshold: int = 100,
        verbose: bool = False
    ) -> bool:
        """Verifies if the root ID has at least `threshold` leaves. This filters out isolated twigs in the dataset."""
        if verbose:
            print(f"Leaves length is {len(self.CG.get_leaves(seg_id, stop_layer=2))} for seg_id {seg_id}", flush=True)
        return len(self.CG.get_leaves(seg_id, stop_layer=2)) >= treshold
    
    def build_old_id_dataset(
        self, 
        ids: list[int], 
        min_l2_size: int = 10,
        show_progress: bool = False,
        verbose: bool = False
    ) -> list[int]:
        """Passes through each root ID and builds a dataset of all past IDs that have at least `min_l2_size` leaves.
        Args:
            ids: list of root IDs
            min_l2_size: minimum number of leaves for a root ID to be included
        Returns:
            root_to_old_seg_ids: dictionary of root IDs to list of old segment IDs
            old_seg_ids: list of old segment IDs indifferentiated by root ID
        """
        root_to_old_seg_ids = {}
        old_seg_ids = []
        cache = set()
        sampled_id_cache = set()
        iterator = tqdm(ids, desc="Acquiring old IDs", unit="id") if show_progress else ids
        for seg_id in iterator:
            if seg_id in cache or not self.verify_size(seg_id, min_l2_size, verbose=verbose):
                continue
            try:
                all_past_ids = self.CG.get_past_ids(seg_id)['past_id_map'][seg_id]
                filtered_ids = [
                    id for id in all_past_ids
                    if self.verify_size(id, min_l2_size, verbose=verbose) and id not in sampled_id_cache
                ]
                if seg_id not in filtered_ids: filtered_ids.append(seg_id)
                if filtered_ids:
                    root_to_old_seg_ids[seg_id] = filtered_ids
                    old_seg_ids.extend(filtered_ids)
                    sampled_id_cache.update(filtered_ids)
            except Exception as e:
                print(f"Error processing seg_id {seg_id}: {e}")
                continue
            if show_progress:
                iterator.set_postfix(dataset_length=len(old_seg_ids))
        return root_to_old_seg_ids, old_seg_ids
    
    def get_old_seg_ids_from_txt_file(
        self, 
        input_path: str, 
        min_l2_size: int = 10,
        show_progress: bool = False,
        verbose: bool = False
    ) -> list[int]:
        """
        Builds a list of segment IDs from a file containing a comma-separated list of IDs. (e.g downloaded from Codex as a txt file)
            Returns a dictionary of root IDs to list of old segment IDs and a list of old segment IDs indifferentiated by root ID.
        """
        with open(input_path, 'r') as file:
            seg_ids = [int(id.strip()) for id in file.read().strip().split(',')]
        root_to_old_seg_ids, old_seg_ids = self.build_old_id_dataset(seg_ids, min_l2_size=min_l2_size, show_progress=show_progress, verbose=verbose)
        return root_to_old_seg_ids, old_seg_ids

    def build_graph_dataset(
        self, 
        seg_ids: list[int], 
        attributes_list: list[str], 
        save_path: str = None, 
        intermediate_save: int = None, 
        show_progress: bool = True,
        verbose: bool = False
    ) -> list[Data]:
        """
        Builds a graph dataset from a list of segment IDs at the L2 level. Parsing the chunkedgraph tables to build the graph dataset.
        Args:
            seg_ids (list[int]): List of segment IDs to include in the dataset.
            attributes_list (list[str]): List of attributes to include in the dataset.
        """
        data = create_dataset(self.client, seg_ids, attributes_list, save_path=save_path, intermediate_save=intermediate_save, show_progress=show_progress, verbose=verbose)
        if save_path is not None:
            torch.save(data, save_path)
            print(f"Dataset saved to {save_path}", flush=True)
        return data

    def associate_error_to_graph_dataset(
        self, 
        data: list[Data], 
        voxel_resolution: np.ndarray = np.array([16, 16, 40]), 
        save_path: str = None, 
        show_progress: bool = True,
        verbose: bool = False
    ) -> list[Data]:
        """
        Takes neurons as graphs and parses their future operations finding error locations found by Proofreaders. Associates error to closest L2 node.

        Args:
            data (list[torch.PyG]): List of PyTorch Geometric graph data objects.
            voxel_resolution (np.ndarray, optional): Resolution of voxels used for error coordinate calculations. 
                                                    Default is [16,16,40] (FlyWire).
            save_path (str, optional): Path to save the processed dataset. Default is None.
            verbose (bool, optional): Whether to print progress details. Default is False.

        Returns:
            list[torch.PyG]: Processed graph dataset with associated error data. Features of each PyG object are: 
        """

        return create_error_dataset_from_graph_list(
            self.client, data, voxel_resolution=voxel_resolution, save_path=save_path, show_progress=show_progress, verbose=verbose
        )

    def build_error_dataset(
        self, 
        seg_ids: list[int], 
        voxel_resolution: np.ndarray = np.array([16, 16, 40]), 
        show_progress: bool = True,
        verbose: bool = False
    ) -> dict[int, np.ndarray]:
        """
        Builds a dataset of the error coordinates for each segment ID by making a forward pass in the graph of operations.
        Args:
            seg_ids: list of segment IDs
            voxel_resolution: voxel resolution of the dataset
            verbose: whether to print verbose output
        Returns:
            dictionary of segment IDs to error features
        """
        seg_id_to_error_features = {}
        iterator = tqdm(seg_ids, desc="Building error dataset", unit="ID") if show_progress else seg_ids
        for seg_id in iterator:
            error_features = build_error_features(client=self.client, data=None, seg_id=seg_id, voxel_resolution=voxel_resolution, verbose=verbose)
            seg_id_to_error_features[seg_id] = error_features
        return seg_id_to_error_features
    
    def get_url_for_visualization(
        self, 
        seg_id: int, 
        error_features: np.ndarray,
        em_data_url: str = "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14",
        segmentation_url: str = "graphene://middleauth+https://prodv1.flywire-daf.com/segmentation/1.0/flywire_public",
        voxel_resolution: np.ndarray = np.array([16,16,40]),
        local_host: bool = False,
        l2_nodes: torch.Tensor = None
    ) -> str:
        """
        Returns a URL for visualization of a given segment ID in neuroglancer.demoappsot or local host.
        Args:
            seg_id: segment ID
            error_features: error features. Numpy array of shape (num_nodes, 6) containing the error features for each node. x,y,z operation type, operation weight, operation id.
            em_data_url: URL of the EM data. Default is the FlyWire public data.
            segmentation_url: URL of the segmentation data. Default is the FlyWire public data.
            voxel_resolution: voxel resolution of the dataset, needed to go back from nm coordinates to voxel coordinates.
            local_host: whether to use the local host
            node_decisions: node decisions. Tensor of shape (num_nodes, 1) containing the node decisions of the model. True or False.
        Returns:
            URL for visualization of the segment ID
        """
        return get_visualization_url(seg_id=seg_id, error_features=error_features, em_data_url=em_data_url, segmentation_url=segmentation_url, voxel_resolution=voxel_resolution, local_host=local_host, l2_nodes=l2_nodes)


    def get_seg_ids_from_supervoxels(
        self, 
        supervoxel_ids: list[int]
    ) -> list[int]:
        """
        Returns a list of the most recent segment IDs from a list of supervoxel IDs.
        Useful since supervoxels are immutable through time (except if resegmented). With this you can acquire the list of most recent segment IDs from a list of supervoxel IDs.
        And then process this to build the graph/error dataset with build_old_id_dataset.
        """
        def get_latest_seg_id(SV):
            try:
                return self.CG.get_root_id(SV)  
            except Exception:
                return None
        seg_ids = [int(seg_id) for SV_ID in supervoxel_ids if (seg_id := get_latest_seg_id(SV_ID)) is not None]
        return seg_ids

    def test_dataset_integrity(self, data: list[Data]):
        """
        Tests the integrity of the dataset by checking the following:
        - No NaN values in any of the elements in the graph data.
        - No 0 values in `Data.x`.

        Args:
            data: list of PyTorch Geometric graph data objects.

        Returns:
            True if the dataset is valid (no NaN, no 0 in `x`, and correct shape relation), False otherwise.
        """
        flag = True
        for i, data in enumerate(data):
            if torch.isnan(data.x).any():
                print(f"Warning: NaN found in Data.x at index {i}")
                flag = False
            
            zero_indices = (data.x[:,:3] == 0).nonzero(as_tuple=True)[0]
            if zero_indices.numel() > 0:
                print(f"Warning: Zero value found in Data.x coordinates for root ID {data.metadata['seg_id']} at rows {zero_indices.tolist()}: {data.x[zero_indices]}")
                flag = False
        
        if flag:
            print("Dataset is valid")
        else:
            print("Dataset is invalid")
        return flag

    def normalize_features(self, 
    data: list[Data], 
    flywire_normalization: bool = True,
    total_mean: torch.Tensor=torch.tensor([5.0366e+05, 2.4488e+05, 1.4929e+05, 5.1604e+08, 7.2761e+06, 1.5032e+02, 4.5299e+01, 4.6807e+04, 1.6421e+04, 8.9097e+03]),
    total_std: torch.Tensor=torch.tensor([1.3346e+05, 6.2122e+04, 5.0383e+04, 3.0599e+09, 2.2970e+07, 1.4053e+02, 3.1322e+01, 5.9969e+04, 2.2565e+04, 1.2393e+04])):
        """
        Normalizes the dataset by making each feature have 0 mean and unit variance.
        """
        if not flywire_normalization:
            total_mean = torch.mean(torch.cat([d.x for d in data], dim=0), dim=0)
            total_std = torch.std(torch.cat([d.x for d in data], dim=0), dim=0)
        for d in data:
            d.x = (d.x - total_mean) / (total_std + 1e-6)
            d.metadata['normalized'] = True
            d.metadata['total_mean'] = total_mean
            d.metadata['total_std'] = total_std
        return data

    def denormalize_features(self, data: list[Data]):
        """
        Denormalizes the dataset by multiplying by the standard deviation and adding the mean.
        """
        for d in data:
            d.x = (d.x * (d.metadata['total_std'] + 1e-6)) + d.metadata['total_mean']
        return data

    def visualize_graph_3d(self, node_features: torch.Tensor, edge_index: torch.Tensor, error_features: torch.Tensor):
        """
        Visualizes the graph in 3D using matplotlib.
        """
        visualize_graph_3d(node_features, edge_index, error_features)


    """
    The next functions are used for Machine Learning on the graph dataset. A default model is provided for inference on Flywire
    """

    def initialize_model(self, in_channels: int, hidden_channels: int, out_channels: int, pretrained_weights_path: str = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the model.
        """
        model = GCNGlobalPredictor3_5(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)
        if pretrained_weights_path is not None:
            model.load_state_dict(torch.load(pretrained_weights_path, map_location=device), strict=False)
        model.to(device)
        return model


    def train_loop(self, train_loader, model, optimizer, criterion, device, lr=0.001, epochs=2000, save_path=None):

        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True)
        criterion = criterion.to(device)

        best_loss = float("inf")

        for epoch in range(epochs):
            model.train()
            total_loss, total_samples = 0, 0
        
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch)
                fault_class = batch.l2_error_weights.squeeze().clip(0, 1).to(torch.int64)

                individual_losses = criterion(output.squeeze(), fault_class.long())

                sample_weights = torch.sqrt(batch.l2_error_weights + 1).squeeze().to(device)
                weighted_loss = (individual_losses * sample_weights).mean()
                weighted_loss.backward()
                optimizer.step()

                total_loss += weighted_loss.item() * batch.num_graphs
                total_samples += batch.num_graphs

            avg_loss = total_loss / total_samples
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}", flush=True)

            scheduler.step(avg_loss)

            if avg_loss < best_loss and save_path:
                best_loss = avg_loss
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved at {save_path} with loss: {best_loss:.4f}", flush=True)

        print("Training complete!")


    def train_model(self, 
        training_data: list[Data], 
        criterion: torch.nn.Module = torch.nn.CrossEntropyLoss(reduction="none", weight=torch.tensor([1.0, 1.0], dtype=torch.float32)),
        validation_data: list[Data] = None, 
        pretrained_weights_path: str = None, 
        save_path: str = None, 
        batch_size: int = 128, 
        lr: float = 0.001, 
        epochs: int = 2000, 
        hidden_channels: int = 256, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        print("------------------------------------------------------------")
        print("#                  Starting training                       #")
        print("------------------------------------------------------------")


        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        if validation_data is not None:
            val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
        else:
            val_loader = None

        model = self.initialize_model(in_channels=10, hidden_channels=hidden_channels, out_channels=2, 
                                pretrained_weights_path=pretrained_weights_path, device=device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.train_loop(train_loader, model, optimizer, criterion, device, lr=lr, epochs=epochs, save_path=save_path)

        if val_loader is not None:
            evaluate_model(model, val_loader, device)
    
        print("------------------------------------------------------------")
        print("#                  Training complete                       #")
        print("------------------------------------------------------------")
        return model

    def model_inference(self, graph_dataset: list[Data], pretrained_weights_path: str, 
        threshold: float = 0.5,
        hidden_channels: int = 256,
        batch_size: int = 32,
        out_channels: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu", evaluate: bool = True):
        """
        Infers the model on a list of data.
        """
        model = self.initialize_model(in_channels=10, hidden_channels=hidden_channels, out_channels=out_channels, pretrained_weights_path=pretrained_weights_path, device=device)
        model.eval()
        test_loader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=False)
        output_list = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = torch.softmax(model(batch), dim=1)
                output_list.append(output)

        
        if evaluate:
            evaluate_model(model, test_loader, device, threshold=threshold) # Requires having the ground truth labels in the dataset as a .l2_error_weights attribute

        output_cat = torch.cat(output_list, dim=0)
        return output_cat[:,1] > threshold