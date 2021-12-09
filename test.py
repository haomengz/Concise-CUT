import torch
import itertools
from model.dataset import XYDataset
import sys
from utils import save_checkpoint, load_checkpoint, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from torchvision.utils import save_image
from model.contrastive_model import ContrastiveModel


def main():
    model = ContrastiveModel()

    dataset = XYDataset(root_X=config.TRAIN_DIR_X, root_Y=config.TRAIN_DIR_Y, transform=transforms)
    val_dataset = XYDataset(root_X=config.VAL_DIR_X, root_Y=config.VAL_DIR_Y, transform=transforms)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    model.load_networks(config.NUM_EPOCHS - 5)
    for idx, data in enumerate(val_loader):
        model.set_input(data)
        model.forward()
        name = config.NAME
        results = model.get_current_visuals()
        for k, v in results.items():
            save_image(v * 0.5 + 0.5, f"saved_images_{name}/test/{k}_{idx}.png")
      
if __name__ == "__main__":
    main()
