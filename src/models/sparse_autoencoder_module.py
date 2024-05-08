from typing import Any, Dict, Tuple

import torch
from torchtyping import TensorType
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.components.sparse_autoencoder import SparseAutoencoderOutput


class SparseAutoencoderLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # for averaging loss across batches
        # TODO - what metrics do they use in the paper?
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_reconstruction_loss = MeanMetric()
        self.val_reconstruction_loss = MeanMetric()
        self.test_reconstruction_loss = MeanMetric()

        self.train_sparsity_loss = MeanMetric()
        self.val_sparsity_loss = MeanMetric()
        self.test_sparsity_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> SparseAutoencoderOutput:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_reconstruction_loss.reset()
        self.val_sparsity_loss.reset()

    def training_step(
        self, batch: TensorType["batch_size", "activation_dim"], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        sae_output = self.forward(batch)

        # update and log metrics
        self.train_loss(sae_output.loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_reconstruction_loss(sae_output.reconstruction_loss)
        self.log("train/reconstruction_loss", self.train_reconstruction_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_sparsity_loss(sae_output.sparsity_loss)
        self.log("train/sparsity_loss", self.train_sparsity_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return sae_output.loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: TensorType["batch_size", "activation_dim"], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        sae_output = self.forward(batch)

        # update and log metrics
        self.val_loss(sae_output.loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_reconstruction_loss(sae_output.reconstruction_loss)
        self.log("val/reconstruction_loss", self.val_reconstruction_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_sparsity_loss(sae_output.sparsity_loss)
        self.log("val/sparsity_loss", self.val_sparsity_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # May want to compute end of validation epoch aggregate metrics here
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        sae_output = self.forward(batch)

        # update and log metrics
        self.test_loss(sae_output.loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.test_reconstruction_loss(sae_output.reconstruction_loss)
        self.log("test/reconstruction_loss", self.test_reconstruction_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.test_sparsity_loss(sae_output.sparsity_loss)
        self.log("test/sparsity_loss", self.test_sparsity_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SparseAutoencoderLitModule(None, None, None, None)
