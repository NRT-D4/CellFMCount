from functools import partial


import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import lightning as pl

from .image_encoder import CellCoder
from data_handler import cellDataset

class LightningViTCounter(pl.LightningModule):
    def __init__(self,train_images_list,val_images_list,test_images_list,config):
        super(LightningViTCounter, self).__init__()
        self.hyperparameters = config
        self.vit_structure = self.hyperparameters['vit_structure'] # Segment Anything Model (SAM) Encoder variant
        self.conv_layers = self.hyperparameters['conv_layers'] # Number of convolutional layers
        self.model_path = self.hyperparameters['model_path'] # Path to the pretrained model
        self.drop_rate = self.hyperparameters['drop_rate'] # Dropout rate
        self.lr = self.hyperparameters['lr'] # Learning Rate
        self.batch_size = self.hyperparameters['batch_size'] # Batch size
        self.train_images_list = train_images_list
        self.val_images_list = val_images_list
        self.test_images_list = test_images_list


        if self.vit_structure.upper() == "SAM-B":
            self.init_vit_b()
        elif self.vit_structure.upper() == "SAM-L":
            self.init_vit_l()
        elif self.vit_structure.upper() == "SAM-H":
            self.init_vit_h()

        self.input_channels = 3
        self.mlp_ratio = 4
        self.qkv_bias = True

        self.prompt_embed_dim = 256

        self.encoder = CellCoder(
            extract_layers = self.extract_layers,
            depth=self.depth,
            embed_dim=self.embed_dim,
            mlp_ratio=self.mlp_ratio,
            num_heads=self.num_heads,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=self.encoder_global_attn_indexes,
            window_size=14,
            out_chans=self.prompt_embed_dim
        )

        self.density_head = self.make_conv_layers()

        self.loss = nn.MSELoss(reduction='sum')

        self.validation_diffs = []
        self.test_diffs = []

    def make_conv_layers(self):

        layers = []
        in_channels = self.prompt_embed_dim
        num_layers = self.conv_layers
        
        while num_layers-1 > 0:
            # add 1x1 convolutional layer
            layers.append(nn.Conv2d(in_channels, in_channels//2, 1))
            in_channels = in_channels//2
            layers.append(nn.ReLU())
            num_layers -= 1

        layers.append(nn.Conv2d(in_channels, 1, 1))

        return nn.Sequential(*layers)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def load_pretrained_encoder(self):
        """Load pretrained SAM encoder from provided path

        Args:
            model_path (str): Path to SAM model
        """
        state_dict = torch.load(str(self.model_path), map_location="cpu")
        image_encoder = self.encoder
        msg = image_encoder.load_state_dict(state_dict, strict=False)
        print(f"Loading checkpoint: {msg}")
        self.encoder = image_encoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.density_head(x[1]) # x[1] is the output of the encoder
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat , y)
        self.log('train_loss', loss,prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat.sum(), y.sum())
        abs_diff = torch.abs(y_hat.sum() - y.sum())
        self.validation_diffs.append(abs_diff)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat.sum(), y.sum())
        abs_diff = torch.abs(y_hat.sum() - y.sum())
        self.test_diffs.append(abs_diff)
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        mae = torch.stack(self.validation_diffs).mean()
        self.log('val_mae', mae, prog_bar=True)
        self.validation_diffs = []

    def on_test_epoch_end(self):
        mae = torch.stack(self.test_diffs).mean()
        self.log('test_mae', mae, prog_bar=True)
        self.test_diffs = []

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        return [optimizer], [scheduler]
    

    def init_vit_b(self):
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        self.encoder_global_attn_indexes = [2, 5, 8, 11]
        self.extract_layers = [3, 6, 9, 12]

    def init_vit_l(self):
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.encoder_global_attn_indexes = [5, 11, 17, 23]
        self.extract_layers = [6, 12, 18, 24]

    def init_vit_h(self):
        self.embed_dim = 1280
        self.depth = 32
        self.num_heads = 16
        self.encoder_global_attn_indexes = [7, 15, 23, 31]
        self.extract_layers = [8, 16, 24, 32]

    def prepare_data(self):
        self.train_dataset = cellDataset(self.train_images_list)
        self.test_dataset = cellDataset(self.test_images_list)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=4)