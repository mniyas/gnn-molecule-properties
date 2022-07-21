"""Experiment-running framework."""
import argparse
import importlib
import warnings

import pytorch_lightning as pl
from torch_geometric import seed_everything

import wandb
from qm_property_predictor import lit_models


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'qm_property_predictor.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--dev_mode", type=bool, default=False)
    parser.add_argument("--seed", type=str, default=920)
    parser.add_argument("--data_class", type=str, default="PyG_QM9")
    parser.add_argument("--model_class", type=str, default="MPNN")
    parser.add_argument("--log_dir", type=str, default="training/logs")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"qm_property_predictor.data.{temp_args.data_class}")

    model_class = _import_class(  # noqa: F841
        f"qm_property_predictor.models.{temp_args.model_class}"
    )

    # Set seed
    seed_everything(temp_args.seed)

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")  # noqa: F841
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/experiment.py --wandb --max_epochs=3 --gpus='0,' --num_workers=4 --model_class=MPNN --data_class=PyG_QM9
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"qm_property_predictor.data.{args.data_class}")
    model_class = _import_class(f"qm_property_predictor.models.{args.model_class}")
    data = data_class(args)
    model = model_class(args=args)

    lit_model_class = lit_models.BaseLitModel
    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(
            args.load_checkpoint, args=args, model=model
        )
    else:
        lit_model = lit_model_class(args=args, model=model)

    logger = pl.loggers.TensorBoardLogger(args.log_dir)
    if args.wandb:
        logger = pl.loggers.WandbLogger()
        logger.watch(model)
        logger.log_hyperparams(vars(args))

    # early_stopping_callback = pl.callbacks.EarlyStopping(
    #     monitor="val_loss", mode="min", patience=10
    # )
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{args.model_class}-{args.target_idx}-{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}",
        monitor="val_loss",
        mode="min",
        dirpath=args.log_dir,
    )
    model_summary_callback = pl.callbacks.ModelSummary(max_depth=-1)
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [
        # early_stopping_callback,
        model_checkpoint_callback,
        model_summary_callback,
        lr_monitor_callback,
    ]
    if args.dev_mode:
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            logger=logger,
            fast_dev_run=True,
            limit_train_batches=0.1,
            limit_val_batches=0.01,
            num_sanity_val_steps=2,
            overfit_batches=0.01,
        )
    else:
        trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)

    # Ignore batch size warning from PL
    warnings.filterwarnings(
        "ignore",
        ".*Trying to infer the `batch_size` from an ambiguous collection.*",
    )
    # pylint: disable=no-member
    trainer.tune(
        lit_model, datamodule=data
    )  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)
    # pylint: enable=no-member
    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path:
        print("Best model saved at:", best_model_path)
        if args.wandb:
            wandb.save(best_model_path)
            print("Best model also uploaded to W&B")


if __name__ == "__main__":
    main()
