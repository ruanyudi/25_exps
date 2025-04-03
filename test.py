import torch
import torchvision
from torch import nn
import lightning as L
from torch.utils.data import DataLoader
from utils import parse_args
from omegaconf import OmegaConf
from models.lightning_base import BaseModule
from datasets.MyDatasets import MyDataset


def test(config, ckpt_path):
    model = BaseModule(config)
    trainer = L.Trainer(
        max_epochs=config.Parameters.max_epochs,
        log_every_n_steps=1,
    )
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((32, 32))]
    )
    test_dataset = MyDataset(transform, config.Datasets.test_path)
    test_loader = DataLoader(
        test_dataset, batch_size=config.Datasets.Batch_size, shuffle=True
    )
    trainer.test(model, test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    args = parse_args()
    config = args.config
    config = OmegaConf.load(config)
    test(config, args.ckpt_path)
