import pytest
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM
from src.data.components.dummy_dataset import DummyDataset
from src.data.components.activation_buffer import ActivationBuffer


@pytest.fixture
def model():
    return GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m-deduped",
        revision='step143000',
        cache_dir='./pythia-70m-deduped/step143000'
    )


@pytest.fixture
def dataloader():
    return DataLoader(DummyDataset(num_samples=1000, vocab_size=100, seq_len=128), batch_size=32)


def test_activation_buffer(model, dataloader):
    batch_size = 16
    buffer_size = 8192
    refill_fraction = 0.5
    activation_dim = 2048
    activation_buffer = ActivationBuffer(
        model,
        dataloader,
        "gpt_neox.layers.1.mlp.act:1",
        batch_size=batch_size,
        buffer_size=buffer_size,
        refill_fraction=refill_fraction
    )
    activation_batch = next(activation_buffer)
    assert activation_batch.shape[0] == batch_size
    assert activation_batch.shape[1] == activation_dim
    assert activation_buffer.num_sampled == batch_size
    assert activation_buffer.num_available == buffer_size - batch_size

    # Run buffer until just before it should refill
    num_batches = 1
    for _ in range(int(buffer_size * refill_fraction // batch_size - 2)):
        next(activation_buffer)
        num_batches += 1
        assert activation_buffer.num_sampled == batch_size * num_batches

    assert activation_buffer.num_available == int(buffer_size * refill_fraction + batch_size)
    assert activation_buffer.num_sampled == int(buffer_size - buffer_size * refill_fraction - batch_size)

    next(activation_buffer)
    assert activation_buffer.num_available == buffer_size
    assert activation_buffer.num_sampled == 0


