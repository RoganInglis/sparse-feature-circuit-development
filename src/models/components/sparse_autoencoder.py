import torch
from typing import Optional
from torchtyping import TensorType
from dataclasses import dataclass
from torch import nn


@dataclass
class SparseAutoencoderOutput:
    x_hat: TensorType["batch", "input_size"]  # Reconstruction of input
    f: TensorType["batch", "num_features"]  # Encoded input
    loss: Optional[TensorType["batch"]] = None
    reconstruction_loss: Optional[TensorType["batch"]] = None
    sparsity_loss: Optional[TensorType["batch"]] = None


class SparseAutoencoder(nn.Module):
    def __init__(self, input_size: int, num_features: int, lambda_: float = 1e-3):
        super(SparseAutoencoder, self).__init__()
        self.lambda_ = lambda_

        self.input_bias = nn.Parameter(torch.zeros(input_size))
        self.encoder = nn.Linear(input_size, num_features, bias=True)
        # TODO - ensure rows of decoder weights are unit vectors?
        self.decoder = nn.Linear(num_features, input_size, bias=False)

    def encode(self, x: TensorType["batch", "input_size"]) -> TensorType["batch", "num_features"]:
        return nn.functional.relu(self.encoder(x - self.input_bias))

    def decode(self, f: TensorType["batch", "num_features"]) -> TensorType["batch", "input_size"]:
        return self.decoder(f) + self.input_bias

    def forward(self, x: TensorType["batch", "input_size"], return_loss: bool = True) -> SparseAutoencoderOutput:
        f = self.encode(x)
        x_hat = self.decode(f)

        loss = None
        reconstruction_loss = None
        sparsity_loss = None
        if return_loss:
            reconstruction_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
            sparsity_loss = self.lambda_ * torch.mean(torch.abs(f))
            loss = reconstruction_loss + sparsity_loss

        return SparseAutoencoderOutput(
            x_hat=x_hat,
            f=f,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss
        )
