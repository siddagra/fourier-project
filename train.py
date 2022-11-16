
def get_data(config):
    from pl_bolts.datamodules import CIFAR10DataModule
    from transforms import val_transforms, train_transforms
    cifar10_dm = CIFAR10DataModule(
        data_dir=config.dataset_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_transforms=train_transforms,
        test_transforms=val_transforms,
        val_transforms=val_transforms,
        pin_memory=False,
        shuffle=False
    )
    return cifar10_dm


def get_model(config):
    from engine import FiT
    return FiT(config)


def main(config):
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
    import warnings
    warnings.filterwarnings("ignore")

    trainer = Trainer(
        accelerator="auto",
        logger=TensorBoardLogger(save_dir=config.save_dir),
        callbacks=[LearningRateMonitor(
            logging_interval="step"), TQDMProgressBar(refresh_rate=1)],
        move_metrics_to_cpu=True, precision=16, amp_backend="native"
    )

    trainer.fit(get_model(config), get_data(config))


if __name__ == '__main__':
    import json
    from types import SimpleNamespace
    with open("config.json", "r") as f:
        config = SimpleNamespace(** json.load(f))
    main(config)
