from importlib import import_module
import argparse
from omegaconf import DictConfig


def instantiate(config: DictConfig, instantiate_module=True):
    """Get arguments from config."""
    module = import_module(config.module_name)
    class_ = getattr(module, config.class_name)
    if instantiate_module:
        init_args = {
            k: v for k, v in config.items() if k not in ["module_name", "class_name"]
        }
        return class_(**init_args)
    else:
        return class_


def parse_args():
    parser = argparse.ArgumentParser(description="model configs")
    # Bug fix: Split the two add_argument calls
    parser.add_argument(
        "--config", default="configs/resnet18.yaml", help="the config file"
    )
    parser.add_argument(
        "--ckpt_path",
        default="./checkpoints/best_model.ckpt",
        help="the checkpoint file",
    )
    return parser.parse_args()
