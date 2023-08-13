import torch
import pytorch_lightning as pl

import config
from customResnet import Model
from dataset import CIFARDataModule

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    print("Train.py initialized...")
    model = Model(
        learning_rate=config.LEARNING_RATE,
        num_classes = config.NUM_CLASSES
    )
    print("Model Constructed")
    
    dm = CIFARDataModule(
        data_dir = config.DATA_DIR,
        batch_size = config.BATCH_SIZE,
        num_workers = config.NUM_WORKERS
    )
    print("Data Model Constructed")

    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
    )
    print("Trainer Object Constructed")

    trainer.fit(model, dm)
    print("Training Model Called")
    
    trainer.test(model, dm)
    print("Testing Model Called")
