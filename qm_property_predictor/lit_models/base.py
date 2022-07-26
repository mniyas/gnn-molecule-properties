import argparse

import numpy as np
import pytorch_lightning as pl
import torch
from torchmetrics import MeanAbsoluteError
from warmup_scheduler import GradualWarmupScheduler

from ..models import MPNN

TARGET_IDX = 1
OPTIMIZER = "Adam"
SCHEDULER = "OneCycleLR"
LR = 1e-3
LOSS = "l1_loss"
ONE_CYCLE_TOTAL_STEPS = 100
EXP_GAMMA = 0.9961697


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


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
        scheduler = self.args.get("lr_scheduler", SCHEDULER)
        if scheduler == "GradualWarmupScheduler":
            self.scheduler_class = GradualWarmupScheduler
        elif scheduler == "CosineWarmupScheduler":
            self.scheduler_class = CosineWarmupScheduler
        else:
            self.scheduler_class = getattr(torch.optim.lr_scheduler, scheduler)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        self.loss_fn = getattr(torch.nn.functional, loss)

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", 1)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)
        self.exponential_gamma = self.args.get("exponential_gamma", EXP_GAMMA)
        self.gw_total_epoch = self.args.get("gw_total_epoch", 1)
        self.cw_warmup = self.args.get("cw_warmup", 100)
        self.cw_max_iters = self.args.get("cw_max_iters", 2000)
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
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            default="OneCycleLR",
            help="LR Scheduler class from torch.optim.lr_scheduler",
        )
        parser.add_argument("--one_cycle_max_lr", type=float, default=1)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--exponential_gamma", type=float, default=EXP_GAMMA)
        parser.add_argument(
            "--gw_total_epoch",
            type=int,
            default=1,
            help="GradualWarmup Scheduler Total Epoch parameter",
        )
        parser.add_argument(
            "--cw_warmup", type=int, default=100, help="CosineWarmup Scheduler Warmup parameter"
        )
        parser.add_argument(
            "--cw_max_iters",
            type=int,
            default=2000,
            help="CosineWarmup Scheduler Max Iter parameter",
        )
        parser.add_argument(
            "--loss",
            type=str,
            default=LOSS,
            help="loss function from torch.nn.functional",
        )
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.scheduler_class == torch.optim.lr_scheduler.OneCycleLR:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.one_cycle_max_lr,
                total_steps=self.one_cycle_total_steps,
            )
        elif self.scheduler_class == GradualWarmupScheduler:
            after_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.exponential_gamma
            )
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1.0,
                total_epoch=self.gw_total_epoch,
                after_scheduler=after_scheduler,
            )
        elif self.scheduler_class == CosineWarmupScheduler:
            scheduler = CosineWarmupScheduler(
                optimizer, warmup=self.cw_warmup, max_iters=self.cw_max_iters
            )
        else:
            return optimizer
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
        return output, batch.y

    def validation_epoch_end(self, validation_step_outputs):
        error = 0
        num_total = 0
        for output, true_val in validation_step_outputs:
            error += (output - true_val).abs().sum().item()
            num_total += output.shape[0]
        val_loss = error / num_total
        self.log(
            "val_mae_weighted",
            val_loss,
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
