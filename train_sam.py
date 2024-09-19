import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from sklearn.model_selection import train_test_split

from model.vit_counter import LightningViTCounter

from glob import glob

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--drop_rate", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--vit_structure", type=str, default="SAM-H")
    parser.add_argument("--conv_layers", type=int, default=3)
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="DCC")
    return parser.parse_args()


def train_model(args):

    # ===== Dataset related tasks =====
    # As per the SAU-Net paper, we will use random splits for each dataset. We won't have separate validation sets.
    # The following datasets will be supported and the splits will be as follows:
    # 1. DCC: 100 images for training, 76 images for testing (43% test split)
    # 2. MBM: 15 images for training, 29 images for testing (66% test split)
    # 3. IDCIA: 250 images for training, 108 images for testing (30% test split)
    if args.dataset == "DCC":
        all_imgs = glob("../Datasets/DCC/*/images/*.png")
        train_images_list, test_images_list = train_test_split(
            all_imgs, test_size=0.43, random_state=42
        )
        validation_images_list = test_images_list

    elif args.dataset == "MBM":
        all_imgs = glob("../Datasets/MBM/*/images/*.png")
        train_images_list, test_images_list = train_test_split(
            all_imgs, test_size=0.65, random_state=42
        )
        validation_images_list = test_images_list

    elif args.dataset == "IDCIA":
        all_imgs = glob("../Datasets/IDCIA/*/images/*.png")
        train_images_list, test_images_list = train_test_split(
            all_imgs, test_size=0.30, random_state=42
        )
        validation_images_list = test_images_list
    else:
        raise ValueError("Dataset not supported")

    print(
        f"Found {len(train_images_list)} training images, {len(validation_images_list)} validation images, and {len(test_images_list)} testing images."
    )

    

    # ===== Model training related tasks =====
    # We will first prepare the config dictionary for the model

    match args.vit_structure:
        case "SAM-H":
            model_path = "weights/sam_vit_h.pth"
        case "SAM-L":
            model_path = "weights/sam_vit_l.pth"
        case "SAM-B":
            model_path = "weights/sam_vit_b.pth"
        case _:
            raise ValueError("Invalid ViT structure")

    config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "drop_rate": args.drop_rate,
        "epochs": args.epochs,
        "model_path": model_path,
        "vit_structure": args.vit_structure,
        "conv_layers": args.conv_layers,
        "freeze": args.freeze,
    }

    # We will now initialize the model
    model = LightningViTCounter(
        config=config,
        train_images_list=train_images_list,
        val_images_list=validation_images_list,
        test_images_list=test_images_list,
    )

    # Load the pretrained model
    model.load_pretrained_encoder()

    # Loggers
    wandb_logger = pl_loggers.WandbLogger(project="SAM-Counter")

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mae",
        mode="min",
        save_top_k=1,
        dirpath=f"ckpts/{wandb_logger.experiment.id}",
        filename="{epoch}-{val_mae:.2f}",
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    # log the config
    wandb.config.update(config)

    # Fit the model
    trainer.fit(model)

    # Load the best model
    model = LightningViTCounter.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        train_images_list=train_images_list,
        val_images_list=validation_images_list,
        test_images_list=test_images_list,
        config=config,
    )

    # Test the model
    trainer.test(model)

    # End the WandB run
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
