import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar, RichModelSummary, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from typing import Callable


class LitAgeEstimationModule(LightningModule):
    def __init__(
        self, 
        model: nn.Module, 
        loss_fn: Callable, 
        learning_rate: float,
        betas: tuple[float, float],
        eps: float,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.save_hyperparameters({'learning_rate': learning_rate, 'betas': betas, 'eps': eps})
        
    def forward_step(self, batch, data_split):
        input, label = batch
        input, label = input.to(self.device), label.to(self.device)

        output = self.model(input)
        loss = self.loss_fn(output, label)

        self.log(f'{data_split}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        self.forward_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self.forward_step(batch, 'valid')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


def train(
    model,
    loss_fn,
    training_data,
    validation_data,
    training_config,
    experiment_folder: str,
    wandb_project: str,
    device,
):
    train_dataloader = DataLoader(training_data, batch_size=training_config.batch_size, shuffle=True)
    valid_dataloader = DataLoader(validation_data, batch_size=training_config.batch_size, shuffle=False)

    age_estimation_module = LitAgeEstimationModule(model, loss_fn, training_config.learning_rate, training_config.betas, training_config.eps)

    wandb_logger = WandbLogger(project=wandb_project)

    callbacks=[
        EarlyStopping(
            monitor='val_loss', 
            patience=training_config.epochs_patience,
            mode='min',
        ),
        ModelCheckpoint(
            dirpath="checkpoints/best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best",
            verbose=True
        ),
        ModelCheckpoint(
            dirpath="checkpoints/epochs",
            filename="epoch{epoch}",  # epoch number will be filled in
            save_top_k=-1,            # save *all* epochs
            every_n_epochs=1,         # every epoch
            save_on_train_epoch_end=True
        )
        RichProgressBar(),
        RichModelSummary(),
    ]

    trainer = Trainer(
        default_root_dir=experiment_folder,
        max_epochs=training_config.n_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        accelerator=device,
        devices=1,
        deterministic=True,
    )

    trainer.fit(
        model=age_estimation_module, 
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )