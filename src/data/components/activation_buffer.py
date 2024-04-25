import torch
import einops
import torchlens as tl
from torchtyping import TensorType
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
"""
An iterator which returns activations from the model. The activations are taken from a specific point in the model.
The activations are stored in a buffer so that we can alternate between training the downstream model and collecting 
the activations. The buffer is filled by running data from a given dataloader through the model. The ActivationBuffer 
should be an iterable which returns a batch of activations ready for training the downstream model.
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActivationBuffer(object):
    def __init__(self, model: nn.Module, dataloader: DataLoader, activation_name: str, batch_size: int = 32,
                 buffer_size: int = 32_000, refill_fraction: float = 0.5, progress_bar: bool = True):
        """
        Args:
            model: The model from which to collect activations
            dataloader: The dataloader to use to fill the buffer
            activation_name: The name of the activation to collect
            batch_size: The batch size of activations to return
            buffer_size: The size of the buffer to store activations
            refill_fraction: The fraction of the buffer that should be filled before returning activations
        """
        self.model = model
        self.dataloader = dataloader
        self.activation_name = activation_name
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.refill_fraction = refill_fraction
        self.progress_bar = progress_bar

        assert self.buffer_size >= self.batch_size, "Buffer size must be greater than or equal to the batch size."

        activations_per_batch = self.dataloader.batch_size * self.dataloader.dataset.seq_len

        assert self.buffer_size >= activations_per_batch, "Buffer size must be greater than or equal to the number of activations per original dataset batch."

        # Need to ensure that when the buffer is refilled we can use an exact number of activation batches
        assert self.buffer_size * (1 - self.refill_fraction) % activations_per_batch == 0, "Buffer refill fraction * buffer size must be a multiple of the number of activations dataloader batch size multiplied by the sequence length."

        if self.buffer_size % activations_per_batch != 0:
            self.buffer_size = activations_per_batch * (self.buffer_size // activations_per_batch)
            print(f"Buffer size must be a multiple of the dataloader batch size multiplied by the sequence length "
                  f"(the number of activations per original dataset batch - {activations_per_batch}) but got "
                  f"{self.buffer_size}. Setting buffer size to {self.buffer_size}.")

        self._activations = None
        self._sampled_activations = None

        self.model.eval()

        self._refill()

    def _init_activations(self, dims: int):
        self._activations = torch.zeros((self.buffer_size, dims), device='cpu')
        self._sampled_activations = torch.ones((self.buffer_size,), device='cpu')

    @property
    def num_available(self):
        if self._sampled_activations is None:
            return 0
        return int(self.buffer_size - self.num_sampled)

    @property
    def num_sampled(self):
        if self._sampled_activations is None:
            return 0
        return int(self._sampled_activations.sum())

    @property
    def filled_fraction(self):
        return self.num_available/self.buffer_size

    def _sample_indices(self, n: int, sampled: bool = False):
        valid_indices = torch.nonzero(self._sampled_activations == int(sampled)).squeeze()
        indices = torch.randperm(valid_indices.shape[0])[:n]
        return valid_indices[indices]

    def _get_activations_from_model(self, batch: TensorType["batch", "seq_len"]):
        model_history = tl.log_forward_pass(self.model, batch, layers_to_save=[self.activation_name])
        activations = model_history[self.activation_name].tensor_contents.cpu()
        activations = einops.rearrange(activations, "b s d -> (b s) d")
        return activations

    def _add_activations_to_buffer(self, activations: TensorType["batch", "dims"]):
        indices_to_replace = self._sample_indices(activations.shape[0], sampled=True)
        self._activations[indices_to_replace] = activations
        self._sampled_activations[indices_to_replace] = 0

    @torch.no_grad()
    def _refill(self):
        self.model.to(device)

        if self.progress_bar:
            progress_bar = tqdm(desc="Filling activation buffer", total=self.buffer_size)
            progress_bar.update(self.num_available)
            progress = lambda x: progress_bar.update(x)
        else:
            progress = lambda x: x

        for data in self.dataloader:
            activations = self._get_activations_from_model(data)

            if self._activations is None:
                self._init_activations(dims=activations.shape[1])

            self._add_activations_to_buffer(activations)
            progress(activations.shape[0])

            if self.filled_fraction >= 1.:
                break
        self.model.to("cpu")

        if self.progress_bar:
            progress_bar.close()

    def __iter__(self):
        return self

    def _sample_activation_batch(self):
        indices = self._sample_indices(self.batch_size)
        self._sampled_activations[indices] = 1
        return self._activations[indices]

    def __next__(self):
        batch = self._sample_activation_batch()

        # Refill the buffer if necessary
        if self.filled_fraction <= self.refill_fraction:
            self._refill()

        return batch
