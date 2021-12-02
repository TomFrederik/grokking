import argparse
from math import factorial

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import wandb

from model import GrokkingTransformer
from datasets import S5Data


def main(
    lr,
    batch_size,
    data_path,
    train_ratio,
    seed,
    steps
):
    # seeding
    pl.seed_everything(seed)

    # data
    data = S5Data(data_path=data_path)
    idcs = np.random.permutation(np.arange(len(data)))
    train_idcs = idcs[:int(train_ratio * len(idcs))]
    val_idcs = idcs[int(train_ratio * len(idcs)):]
    train_loader = DataLoader(Subset(data, train_idcs), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(data, val_idcs), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model
    model = GrokkingTransformer(num_tokens=factorial(5))

    # training
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_steps=steps,
        checkpoint_callback=pl.callbacks.model_checkpoint.ModelCheckpoint(
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
            prefix="",
        ),
        logger=pl.loggers.WandbLogger(project="grokking-transformer", config={"lr": lr, "batch_size": batch_size, "steps": steps}),
        progress_bar_refresh_rate=1,
    )

    trainer.fit(model, train_loader, val_loader)
    print('hi 6')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_path", type=str, default="./data/s5.npy")
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10**5)
    
    main(**vars(parser.parse_args()))