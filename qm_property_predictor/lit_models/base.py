import argparse

import pytorch_lightning as pl
import torch
from torchmetrics import MeanAbsoluteError

from ..models import MPNN

TARGET_IDX = 1
OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "l1_loss"
ONE_CYCLE_TOTAL_STEPS = 100


class BaseLitModel(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning base class.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        self.target_idx = self.args.get("target_idx", TARGET_IDX)
        # All models except MPNN uses the atomization energy for targets U0, U, H, and G.
        if not isinstance(model, MPNN) and self.target_idx in [7, 8, 9, 10]:
            self.target_idx = self.target_idx + 5
        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        self.loss_fn = getattr(torch.nn.functional, loss)

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)
        # TODO: Evaluate MAE calculation method
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--optimizer",
            type=str,
            default=OPTIMIZER,
            help="optimizer class from torch.optim",
        )
        parser.add_argument("--target_idx", type=int, default=TARGET_IDX)
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument(
            "--loss",
            type=str,
            default=LOSS,
            help="loss function from torch.nn.functional",
        )
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.one_cycle_max_lr,
            total_steps=self.one_cycle_total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    # TODO: Evaluate forward method
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        output = self(batch)
        loss = self.loss_fn(output, batch.y[:, self.target_idx].unsqueeze(1))
        self.log("train_loss", loss)
        self.train_mae(output, batch.y[:, self.target_idx].unsqueeze(1))
        self.log("train_mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        output = self(batch)
        loss = self.loss_fn(output, batch.y[:, self.target_idx].unsqueeze(1))
        self.log("val_loss", loss, prog_bar=True)
        self.val_mae(output, batch.y[:, self.target_idx].unsqueeze(1))
        self.log(
            "val_mae",
            self.val_mae,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        output = self(batch)
        self.test_mae(output, batch.y[:, self.target_idx].unsqueeze(1))
        self.log(
            "test_mae",
            self.test_mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
