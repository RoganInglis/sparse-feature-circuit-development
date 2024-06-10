import numpy as np
import torch
from src.feature_matching import get_full_cfg, get_sae, get_model, get_dataset, get_activations, match_features

from feature_visualisation import visualise_features_for_sae_path

torch.set_grad_enabled(False)

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    checkpoint_paths = [
        '/home/rogan/git_repos/sparse-feature-circuit-development/scripts/checkpoints/p0hcxgno/final_400003072',
        '/home/rogan/git_repos/sparse-feature-circuit-development/scripts/checkpoints/nt6053cl/final_400003072',
        '/home/rogan/git_repos/sparse-feature-circuit-development/scripts/checkpoints/dhqyxuqo/final_400003072',
        '/home/rogan/git_repos/sparse-feature-circuit-development/scripts/checkpoints/9uxbhb8i/final_400003072',
        '/home/rogan/git_repos/sparse-feature-circuit-development/scripts/checkpoints/e30haena/final_400003072',
        '/home/rogan/git_repos/sparse-feature-circuit-development/scripts/checkpoints/36nxoute/final_400003072',
        '/home/rogan/git_repos/sparse-feature-circuit-development/scripts/checkpoints/03gvobe4/final_400003072',
        '/home/rogan/git_repos/sparse-feature-circuit-development/scripts/checkpoints/so58immk/final_400003072',
        '/home/rogan/git_repos/sparse-feature-circuit-development/scripts/checkpoints/03sw4xom/final_400003072',
        '/home/rogan/git_repos/sparse-feature-circuit-development/scripts/checkpoints/ieqwggci/final_400003072',
    ]
    batch_size = 32
    total_tokens = 640000
    sparsity_threshold = 1e-6

    activation_data = dict()
    for checkpoint_path in checkpoint_paths[:2]:
        full_cfg = get_full_cfg(checkpoint_path)
        sae = get_sae(checkpoint_path, full_cfg)
        model = get_model(sae, full_cfg)
        dataset = get_dataset(full_cfg)
        activations = get_activations(
            sae,
            model,
            dataset,
            batch_size=batch_size,
            total_tokens=total_tokens,
            sparsity_threshold=sparsity_threshold
        )
        activation_data[checkpoint_path] = {
            'activations': activations,
            'cfg': full_cfg,
            'sae': sae,
            'model': model
        }

    indices_orig, indices_matched, matched_distances = match_features(
        activation_data[checkpoint_paths[0]]['activations'],
        activation_data[checkpoint_paths[1]]['activations']
    )

    # Visualise the most closely matched features
    ind = np.argsort(matched_distances)
    ind = ind[:10]
    top_indices_orig = indices_orig[ind].tolist()
    top_indices_matched = indices_matched[ind].tolist()

    visualise_features_for_sae_path(checkpoint_paths[0], top_indices_orig)
    visualise_features_for_sae_path(checkpoint_paths[1], top_indices_matched)


if __name__ == "__main__":
    main()
