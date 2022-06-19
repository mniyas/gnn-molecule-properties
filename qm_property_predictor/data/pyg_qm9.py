"""QM9 DataModule from Torch Geometric"""
import argparse

from torch.utils.data import random_split
from torch_geometric.datasets import QM9

from qm_property_predictor.data.base_data_module import BaseDataModule

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"


class PyG_QM9(BaseDataModule):
    """
    PyG_QM9 DataModule. The QM9 data pre-processed by PyG team.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.data_dir = DOWNLOADED_DATA_DIRNAME

    def setup(self, stage=None) -> None:
        """Split into train, val, test, and set dims."""
        qm9_full = QM9(self.data_dir)
        self.data_train, self.data_val, self.data_test = random_split(qm9_full, [110000, 10831, 10000])  # type: ignore
