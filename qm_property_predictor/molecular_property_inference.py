import argparse
import os.path as osp

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNetPlusPlus

target_map = {
    0: "mu",
    1: "alpha",
    2: "homo",
    3: "lumo",
    4: "gap",
    5: "r2",
    6: "zpve",
    7: "u0",
    8: "u298",
    9: "h298",
    10: "g298",
    11: "cv",
}


class MoleculeProperty:
    """Class to predict the properties of molecules"""

    def __init__(self):
        path = osp.join("..", "data")
        dataset = QM9(path)
        idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
        dataset.data.y = dataset.data.y[:, idx]
        self.dataset = dataset
        self.models = {}
        for target in range(12):
            if target == 4:
                continue
            model = DimeNetPlusPlus.from_qm9_pretrained(path, self.dataset, target)[0]
            model.eval()
            self.models[target] = model

    @torch.no_grad()
    def predict(self, data) -> str:
        """Predict/infer text in input image (which can be a file path)."""
        predictions = []
        for target in range(12):
            if target == 4:
                continue
            pred = self.models[target](data.z, data.pos, data.batch)
            predictions.append((target_map[target], pred.item()))
        return predictions

    def access_data(self):
        return self.dataset


if __name__ == "__main__":
    MoleculeProperty()
