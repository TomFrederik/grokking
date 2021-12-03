import argparse
import logging
from math import factorial
import os

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
    heads,
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
    verbose,
    log_freq,
    num_workers,
    disable_logging,
    checkpoints
):
    # set wandb logging mode
    if disable_logging:
        os.environ['WANDB_MODE'] = 'offline'
    else:
        os.environ['WANDB_MODE'] = 'online'
    
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
    train_loader = DataLoader(Subset(data, train_idcs), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(Subset(data, val_idcs), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # model
    optim_kwargs = {
        'lr': lr,
        'weight_decay':weight_decay,
        'betas': (beta1, beta2)
    }
    model_kwargs = {
        'heads':heads,
        'layers':layers,
        'width':width,
        'num_tokens':factorial(num_elements),
        'optim_kwargs':optim_kwargs,
        'checkpoints':checkpoints
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
    callbacks = []
    callbacks.append(
        pl.callbacks.model_checkpoint.ModelCheckpoint(
            save_top_k=1,
            verbose=verbose,
            monitor="Validation/Loss",
            mode="min"
        )
    )
    callbacks.append(pl.callbacks.progress.TQDMProgressBar(refresh_rate=log_freq))
    # callbacks.append(pl.callbacks.ModelSummary(max_depth=5))
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_steps=steps,
        log_every_n_steps=log_freq,
        callbacks=callbacks,
        logger=pl.loggers.WandbLogger(project="grokking-transformer", config=config),
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
    parser.add_argument("--heads", type=int, default=4)
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
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disable_logging", action="store_true")
    parser.add_argument("--checkpoints", type=int, default=None, nargs="*", help="List of number of steps after which to save model.")
    
    main(**vars(parser.parse_args()))
