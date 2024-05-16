import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as pl

from model.vit_counter import LightningViTCounter

from glob import glob

def main():
    model = LightningViTCounter(
        model_path="weights/sam_vit_b.pth", # Path to the pretrained model
        vit_structure="SAM-B",
        drop_rate=0,
        images_list=glob("../Datasets/DCC/trainval/images/*.png"),
    )

    model.load_pretrained_encoder()

    # Freeze the encoder
    model.freeze_encoder()

    print(f"Everything is ready! Start training...")

    # train
    trainer = pl.Trainer(max_epochs=100, accelerator='auto')
    trainer.fit(model)



if __name__ == "__main__":
    main()