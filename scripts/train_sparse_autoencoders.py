import torch
import numpy as np
from transformer_lens.loading_from_pretrained import PYTHIA_CHECKPOINTS
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_checkpoint_indices(num_checkpoints: int = 5):
    end_ckpt_steps = 143_000
    ckpt_steps = np.linspace(0, end_ckpt_steps, num_checkpoints).round(-3).astype(int).tolist()
    return [PYTHIA_CHECKPOINTS.index(ckpt_step) for ckpt_step in ckpt_steps]


def main():
    num_checkpoints = 10
    checkpoint_indices = get_checkpoint_indices(num_checkpoints)
    checkpoint_indices = [2*(i + 1) for i in range(10)]
    #checkpoint_indices = [x for x in get_checkpoint_indices(10) if x not in checkpoint_indices]
    #checkpoint_indices = checkpoint_indices[-1:]

    for checkpoint_index in checkpoint_indices:
        cfg = LanguageModelSAERunnerConfig(
            # Data generating function
            model_name='pythia-70m-deduped',
            hook_name="blocks.3.hook_resid_pre",
            hook_layer=3,
            d_in=512,
            dataset_path='apollo-research/sae-monology-pile-uncopyrighted-tokenizer-EleutherAI-gpt-neox-20b',
            is_dataset_tokenized=True,
            model_from_pretrained_kwargs=None if checkpoint_index is None else {'checkpoint_index': checkpoint_index},
            # SAE Parameters
            mse_loss_normalization=None,
            expansion_factor=16,
            b_dec_init_method="geometric_median",
            # Training Parameters
            lr=5e-5,
            lr_scheduler_name="constant",
            lr_warm_up_steps=0,
            l1_warm_up_steps=10_000,
            l1_coefficient=3,
            normalize_sae_decoder=False,
            decoder_heuristic_init=True,
            init_encoder_as_decoder_transpose=True,
            scale_sparsity_penalty_by_decoder_norm=True,
            lp_norm=1.,
            train_batch_size_tokens=4096,
            context_size=512,
            # Activation Store Parameters
            n_batches_in_buffer=64,
            training_tokens=1_000_000 * 400,
            store_batch_size_prompts=16,
            # Wandb
            log_to_wandb=True,
            wandb_project="pythia-sae",
            wandb_log_frequency=10,
            # Misc
            device=device,
            seed=42,
            n_checkpoints=1,
            checkpoint_path="checkpoints",
            dtype="float32"
        )
        SAETrainingRunner(cfg).run()


if __name__ == "__main__":
    main()
