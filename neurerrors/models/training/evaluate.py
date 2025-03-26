if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from neurerrors.models.training.utils.evaluation_metrics import confusion_matrix, calculate_metrics
import torch
import numpy as np

def evaluate_model(model, val_loader, device, threshold: float = 0.5):
    model.eval()
    true_labels, predicted_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = torch.softmax(model(batch), dim=1)
            fault_labels = batch.l2_error_weights.squeeze().clip(0, 1).to(torch.int64)
            
            predicted_classes = (output[:, 1] > threshold).long()

            true_labels.extend(fault_labels.cpu().numpy())
            predicted_labels.extend(predicted_classes.cpu().numpy())

    conf_matrix = confusion_matrix(np.array(true_labels), np.array(predicted_labels), 2)
    precision, recall, f1_scores = calculate_metrics(conf_matrix)

    print(f"Validation Confusion Matrix:\n{conf_matrix} for threshold {threshold}")
    for i in range(2):
        print(f"Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-score={f1_scores[i]:.4f}")
