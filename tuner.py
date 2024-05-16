import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb

from model.vit_counter import LightningViTCounter

from glob import glob

def train_model():
    wandb.init(project="SAM-Counter")
    config = wandb.config
    wandb_logger = pl_loggers.WandbLogger()

    train_images_list = glob("../Datasets/DCC/trainval/images/*.png")
    val_images_list = glob("../Datasets/DCC/test/images/*.png")
    test_images_list = glob("../Datasets/DCC/test/images/*.png")

    model = LightningViTCounter(
        train_images_list=train_images_list,
        val_images_list=val_images_list,
        test_images_list=test_images_list,
        config=config
    )

    model.load_pretrained_encoder()

    if config['freeze']:
        model.freeze_encoder()
        

    wandb_logger.watch(model,log_graph=False)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_mae',
        mode='min',
        save_top_k=1,
        dirpath=f'ckpts/{wandb.run.id}',
        filename='{epoch}-{val_mae:.2f}'
    )

    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=200,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model)

    # load best model
    model = LightningViTCounter.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,train_images_list=train_images_list,val_images_list=val_images_list,test_images_list=test_images_list,config=config)

    trainer.test(model)



if __name__ == "__main__":
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'test_mae',
            'goal': 'minimize'
        },
        'parameters': {
            'lr': {
                'values': [1e-4, 1e-5, 1e-6, 1e-7]
            },
            'batch_size': {
                'values': [2,8],
            },
            'conv_layers': {
                'values': [1,2,3]
            },
            'drop_rate': {
                'values': [0]
            },
            'vit_structure': {
                'values': ['SAM-H']
            },
            'model_path': {
                'values': ['weights/sam_vit_h.pth']
            },
            'freeze':{
                'values': [True,False]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="SAM-Counter")
    wandb.agent(sweep_id, function=train_model,count = 50)

