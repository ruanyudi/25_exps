import lightning
import torch
from torchmetrics.classification import Accuracy
import torch.optim as optim
import torch.nn as nn
from lightning import LightningModule
import torch.nn.functional as F
from torchcam.methods import SmoothGradCAMpp
from utils import instantiate


class BaseModule(LightningModule):
    def __init__(self, config):
        super(BaseModule, self).__init__()
        self.backbone = instantiate(config.Backbone, instantiate_module=False)()
        self.classifier = instantiate(config.Classifier, instantiate_module=False)(
            config
        )
        self.learning_rate = config.Parameters.learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = config.Classifier.num_classes
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        acc = self.train_acc(logits, labels)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        acc = self.val_acc(logits, labels)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        val_accuracy = self.val_acc.compute()
        self.log("val_acc", val_accuracy)
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        acc = self.test_acc(logits, labels)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def on_test_end(self):
        test_accuracy = self.test_acc.compute()
        self.test_acc.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            lr=self.learning_rate,
        )
        return optimizer

    def configure_callbacks(self):
        checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
            monitor="val_acc",
            dirpath="./checkpoints",
            filename="best_model",
            save_top_k=1,
            mode="max",
        )
        return [checkpoint_callback]
