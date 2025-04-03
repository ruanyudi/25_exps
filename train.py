import torch
import torchvision
from torch import nn
import lightning as L
from torch.utils.data import DataLoader
from utils import parse_args
from omegaconf import OmegaConf
from models.lightning_base import BaseModule
from datasets.MyDatasets import MyDataset


def train(config):
    model = BaseModule(config)
    trainer = L.Trainer(
        max_epochs=config.Parameters.max_epochs,
        log_every_n_steps=1,
    )
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((32, 32))]
    )
    train_dataset = MyDataset(transform, config.Datasets.train_path)
    val_dataset = MyDataset(transform, config.Datasets.val_path)
    train_loader = DataLoader(
        train_dataset, batch_size=config.Datasets.Batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.Datasets.Batch_size, shuffle=False
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    args = parse_args()
    config = args.config
    config = OmegaConf.load(config)
    train(config)
