from pytorch_lightning import LightningModule
from FiT import FNet
import torch.nn.functional as F
import torch
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy


class FiTLit(LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters()
        self.config = config
        self.model = FNet(config)

    def forward(self, x):
        _, out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters()
        )
        steps_per_epoch = 45000 // self.config.batch_size
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.config.lr,
                epochs=self.config.epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def on_train_end(self):
        update_bn(self.trainer.datamodule.train_dataloader(),
                  self.swa_model, device=self.device)
