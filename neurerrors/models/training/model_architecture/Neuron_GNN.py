import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch.utils.checkpoint as checkpoint
from torch_geometric.data import Data


class GCNGlobalPredictor3_5(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, pooling="add", dropout=0.2):
        super(GCNGlobalPredictor3_5, self).__init__()
        
        # Initial MLP for node feature transformation
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )
        
        # GCN layers with normalization and residual connections
        self.conv1 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.conv3 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.conv4 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.conv5 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.conv6 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.conv7 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.layer_norm1 = nn.LayerNorm(hidden_channels)
        self.layer_norm2 = nn.LayerNorm(hidden_channels)
        self.layer_norm3 = nn.LayerNorm(hidden_channels)
        self.layer_norm4 = nn.LayerNorm(hidden_channels)
        self.layer_norm5 = nn.LayerNorm(hidden_channels)
        self.layer_norm6 = nn.LayerNorm(hidden_channels)
        self.layer_norm7 = nn.LayerNorm(hidden_channels)
        
        # Global MLP layers for initial and refined global representations
        # Global MLP layers for initial and refined global representations
        self.global_mlp_init = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        # Combined MLP for final node classification
        self.combined_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),  # Dropout for regularization
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),  # Dropout for regularization
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),  # Dropout for regularization
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )
        self.conv_end1 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.layer_norm_end1 = nn.LayerNorm(hidden_channels)
        self.conv_end2 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.layer_norm_end2 = nn.LayerNorm(hidden_channels)

        self.node_classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        
        # Pooling method
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "add":
            self.pool = global_add_pool
        else:
            raise ValueError("Unsupported pooling method. Choose 'mean' or 'add'.")

    def forward(self, data, return_intermediate=False):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        if batch is None or batch.dim() == 0:  # Single graph case
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.int64)  
        # Initial global representation
        global_init = self.pool(x, batch)
        global_repr_init = self.global_mlp_init(global_init)
        global_repr_expanded_init = global_repr_init[batch] # Broadcast global representation per node


        x = self.node_mlp(x)
        
        # GCN Layers with Layer Norm and Residual Connections
        x = self._apply_gcn_layer(self.conv1, x, edge_index, self.layer_norm1)
        x = self._apply_gcn_layer(self.conv2, x, edge_index, self.layer_norm2)
        x = self._apply_gcn_layer(self.conv3, x, edge_index, self.layer_norm3)
        x = self._apply_gcn_layer(self.conv4, x, edge_index, self.layer_norm4)
        x = self._apply_gcn_layer(self.conv5, x, edge_index, self.layer_norm5)
        x = self._apply_gcn_layer(self.conv6, x, edge_index, self.layer_norm6)
        x = self._apply_gcn_layer(self.conv7, x, edge_index, self.layer_norm7)
        # Final global representation
        global_repr = self.pool(x, batch)
        global_repr = self.global_mlp(global_repr)
        global_repr_expanded = global_repr[batch]  # Broadcast final global representation

        #print("Global repr expanded is of shape: ", global_repr_expanded.shape)
        #print("Global repr expanded init is of shape: ", global_repr_expanded_init.shape)
        #print("X is of shape: ", x.shape)
        # Concatenate initial and final global representations with node features
        x = torch.cat([x, global_repr_expanded_init, global_repr_expanded], dim=-1)

        # Combined MLP and node classifier
        x = self.combined_mlp(x)

        x = self._apply_gcn_layer(self.conv_end1, x, edge_index, self.layer_norm_end1)
        x = self._apply_gcn_layer(self.conv_end2, x, edge_index, self.layer_norm_end2)
        
        node_logits = self.node_classifier(x)

        if return_intermediate:
            return node_logits, x
        return node_logits
    
    def _apply_gcn_layer(self, conv, x, edge_index, layer_norm):
        """Helper function to apply a GCN layer with residual connections."""
        residual = x
        x = conv(x, edge_index)
        x = layer_norm(x + residual)  # Residual connection
        return F.relu(x)

    def get_edge_features(self, x, edge_index, attention_weights=None):
        """Optionally add attention weights for edge feature extraction."""
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=-1)
        
        if attention_weights is not None:
            edge_features = edge_features * attention_weights.unsqueeze(-1)
        
        return edge_features