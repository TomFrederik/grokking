import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from x_transformers import TransformerWrapper, Decoder

class GrokkingTransformer(pl.LightningModule):
    def __init__(self, layers=2, width=128, heads=4, num_tokens=7, max_seq_len=5, optim_kwargs=None, checkpoints=None):
        super().__init__()
        self.save_hyperparameters()
        if optim_kwargs is None:
            self.optim_kwargs = {
                'lr': 1e-3,
                'weight_decay':1,
                'betas': (0.9, 0.98),
            }
        else:
            self.optim_kwargs = optim_kwargs
        if checkpoints is None:
            self.checkpoints = []
        else:
            self.checkpoints = checkpoints
        
        self.model = TransformerWrapper(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(
                dim=width,
                heads=heads,
                depth=layers 
            )
        )
        
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x[:,:-1])
        loss = self.loss(y_hat[:,-1], x[:,-1])
        acc = (torch.argmax(y_hat[:,-1], dim=1) == x[:,-1]).sum() / x.shape[0]
        self.log('Training/Accuracy', acc, on_step=True)
        self.log('Training/Loss', loss, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x[:,:-1])
        loss = self.loss(y_hat[:,-1], x[:,-1])
        acc = (torch.argmax(y_hat[:,-1], dim=1) == x[:,-1]).sum() / x.shape[0]
        self.log('Validation/Accuracy', acc, on_epoch=True)
        self.log('Validation/Loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), **self.optim_kwargs)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(step/10, 1))
        return {
            'optimizer': self.optimizer, 
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'frequency': 1,
                'interval': 'step'
                }
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # save model if global step is in checkpoint list
        if self.global_step in self.checkpoints:
            logging.info(f'Saving model after {self.global_step} steps')
            self.trainer.save_checkpoint(os.path.join(f'{wandb.run.dir}', f'{self.global_step}.ckpt'))
