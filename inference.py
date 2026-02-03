import torch
import numpy as np
from torch.utils.data import DataLoader
from model import SimpleCNN
from model import DermatologyDataset

def evaluate_model(model_path, test_npz_path, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    data = np.load(test_npz_path)
    x_test = data["x_val"]
    y_test = data["y_val"].squeeze()

    x_test = x_test.astype("float32") / 255.0

    test_ds = DermatologyDataset(x_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Load model
    model = SimpleCNN(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc
