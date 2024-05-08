import torch
from dataclasses import dataclass
from sae_lens import LanguageModelSAERunnerConfig
from sae_lens.training.lm_runner import language_model_sae_runner


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RunConfig:
    model_name: str
    hook_point: str
    hook_point_layer: int
    d_in: int


def main():
    run_configs = [
        RunConfig(
            model_name='pythia-70m-deduped',
            hook_point="blocks.4.hook_mlp_out",
            hook_point_layer=4,
            d_in=512
        )
    ]

    sparse_autoencoder_dictionaries = []
    for run_config in run_configs:
        cfg = LanguageModelSAERunnerConfig(
            # Data generating function
            model_name=run_config.model_name,
            hook_point=run_config.hook_point,
            hook_point_layer=run_config.hook_point_layer,
            d_in=run_config.d_in,
            dataset_path='EleutherAI/pythia_deduped_pile_idxmaps',
            is_dataset_tokenized=True,
            # SAE Parameters
            mse_loss_normalization=None,
            expansion_factor=16,
            b_dec_init_method="geometric_median",
            # Training Parameters
            lr=8e-4,
            lr_scheduler_name="constant",
            lr_warm_up_steps=10_000,
            l1_coefficient=1e-3,
            lp_norm=1.,
            train_batch_size=4096,
            context_size=512,
            # Activation Store Parameters
            n_batches_in_buffer=64,
            training_tokens=1_000_000 * 50,
            store_batch_size=16,
            # Resampling protocol
            use_ghost_grads=False,
            feature_sampling_window=1000,
            dead_feature_window=1000,  # Not used as use_ghost_grads=False
            dead_feature_threshold=1e-4,  # Not used as use_ghost_grads=False
            # Wandb
            log_to_wandb=True,
            wandb_project="pythia-sae",
            wandb_log_frequency=10,
            # Misc
            device=device,
            seed=42,
            n_checkpoints=0,
            checkpoint_path="checkpoints",
            dtype=torch.float32,
            # model_from_pretrained_kwargs = {'checkpoint_index': 0}
        )
        sparse_autoencoder_dictionary = language_model_sae_runner(cfg)
        sparse_autoencoder_dictionaries.append(sparse_autoencoder_dictionary)
    print(sparse_autoencoder_dictionaries)


if __name__ == "__main__":
    #main()

    from datasets import load_dataset
    dataset = load_dataset("EleutherAI/pythia_deduped_pile_idxmaps", streaming=True)
    print(next(dataset))
