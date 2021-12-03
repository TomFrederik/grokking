import argparse
import logging
from math import factorial

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import wandb

from model import GrokkingTransformer
from datasets import get_dataset


def main(
    lr,
    weight_decay,
    beta1,
    beta2,
    num_heads,
    layers,
    width,
    data_name,
    num_elements,
    data_dir,
    force_data,
    batch_size,
    steps,
    train_ratio,
    seed,
    verbose
):
    # set logging level
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # seeding
    pl.seed_everything(seed)

    # data
    data = get_dataset(descr=data_name, num_elements=num_elements, data_dir=data_dir, force_data=force_data)
    idcs = np.random.permutation(np.arange(len(data)))
    train_idcs = idcs[:int(train_ratio * len(idcs))]
    val_idcs = idcs[int(train_ratio * len(idcs)):]
    train_loader = DataLoader(Subset(data, train_idcs), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(data, val_idcs), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model
    optim_kwargs = {
        'lr': lr,
        'weight_decay':weight_decay,
        'betas': (beta1, beta2)
    }
    model_kwargs = {
        'num_heads':num_heads,
        'layers':layers,
        'width':width,
        'num_tokens':factorial(5),
        'optim_kwargs':optim_kwargs
    }
    model = GrokkingTransformer(**model_kwargs)

    # training
    config = dict(
        **optim_kwargs, 
        **model_kwargs, 
        batch_size=batch_size, 
        steps=steps, 
        data_name=data_name, 
        num_elements=num_elements
    )
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_steps=steps,
        callbacks=[pl.callbacks.model_checkpoint.ModelCheckpoint(
            save_top_k=1,
            verbose=verbose,
            monitor="Validation/Loss",
            mode="min",
        )],
        logger=pl.loggers.WandbLogger(project="grokking-transformer", config=config),
        progress_bar_refresh_rate=1,
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # optimizer args
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)

    # model args
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=128)

    # data args
    parser.add_argument("--data_name", type=str, default="perm_xy", choices=[
        # Permutation
        "perm_xy", 
        "perm_xyx1",
        "perm_xyx",
        
        # Arithmetic, all operations are mod p
        "plus", # x + y
        "minus", # x - y
        "div", # x / y
        "div_odd", # x / y if y is odd else x - y
        "x2y2", # x^2 + y^2
        "x2xyy2", # x^2 + y^2 + xy
        "x2xyy2x", # x^2 + y^2 + xy + x
        "x3xy", # x^3 + y
        "x3xy2y" # x^3 + xy^2 + y
    ])
    parser.add_argument("--num_elements", type=int, default=5) # choose 5 for permutation data, 97 for arithmetic data
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--force_data", action="store_true", help="Whether to force dataset creation.")
    
    # training args
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10**5)
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    
    main(**vars(parser.parse_args()))
