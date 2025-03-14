if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from neurerrors.models.training.model_architecture.Neuron_GNN import GCNGlobalPredictor3_5
from neurerrors.models.training.evaluate import evaluate_model
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GraphDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.num_graphs = len(data)

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        return self.data[idx]

def load_data(train_path, val_path):
    try:
        if val_path is None:
            train_data = torch.load(train_path, weights_only=False)
            return GraphDataset(train_data), None
        else:
            train_data = torch.load(train_path, weights_only=False)
            val_data = torch.load(val_path, weights_only=False)
            return GraphDataset(train_data), GraphDataset(val_data)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def initialize_model(in_channels, hidden_channels, out_channels, model_path=None, device="cpu"):
    model = GCNGlobalPredictor3_5(in_channels, hidden_channels, out_channels, dropout=0.3, pooling="add").to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    return model


def train_model(train_loader, model, device, lr=0.001, epochs=2000, save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss(reduction="none", weight=torch.tensor([1.0, 1.0], dtype=torch.float32).to(device))

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


def main():
    parser = argparse.ArgumentParser(description="Train a GNN model for graph-based classification.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset (.pt)")
    parser.add_argument("--val_path", type=str, default=None, help="Path to the validation dataset (.pt)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pre-trained model (optional)")
    parser.add_argument("--save_path", type=str, default="neurerrors/models/training/weights/default_save.pt", help="Path to save the best model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--hidden_channels", type=int, default=256, help="Number of hidden channels in the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("------------------------------------------------------------")
    print("#                  Starting training                        ")
    print("------------------------------------------------------------")

    train_dataset, val_dataset = load_data(args.train_path, args.val_path)
    if train_dataset is None:
        print("Error: No training data provided")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        val_loader = None

    model = initialize_model(in_channels=10, hidden_channels=args.hidden_channels, out_channels=2, 
                             model_path=args.model_path, device=device)
    train_model(train_loader, model, device, lr=args.lr, epochs=args.epochs, 
                save_path=args.save_path)

    if val_loader is not None:
        evaluate_model(model, val_loader, device)


if __name__ == "__main__":
    main()