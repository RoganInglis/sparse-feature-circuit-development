import numpy as np
import os
import json
import einops
import torch
from datasets import load_dataset, IterableDataset
from transformer_lens import HookedTransformer
from scipy.optimize import linear_sum_assignment
from sae_lens import SAE
from tqdm import tqdm

torch.set_grad_enabled(False)

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"


def get_full_cfg(sae_path: str):
    with open(os.path.join(sae_path, "cfg.json")) as f:
        return json.load(f)


def get_sae(sae_path: str, full_cfg: dict):
    sae = SAE.load_from_pretrained(
        path=sae_path,
        device=device
    )

    # If we trained an Anthropic April-update-style model, we need to fold the decoder norm into the sparsity penalty
    # in order to compute activations correctly.
    if full_cfg['scale_sparsity_penalty_by_decoder_norm']:
        sae.fold_W_dec_norm()

    return sae


def get_model(sae, full_cfg):
    model_from_pretrained_kwargs = full_cfg.get('model_from_pretrained_kwargs') or {}

    return HookedTransformer.from_pretrained(sae.cfg.model_name, device=device, **model_from_pretrained_kwargs)


def get_dataset(full_cfg):
    dataset = load_dataset(
        path=full_cfg['dataset_path'],
        split="train",
        streaming=True,
    )
    return dataset


def get_model_activations(model: HookedTransformer, batch_tokens: torch.tensor, sae_cfg):
    return model.run_with_cache(
        batch_tokens,
        names_filter=[sae_cfg.hook_name],
        stop_at_layer=sae_cfg.hook_layer + 1,
        prepend_bos=sae_cfg.prepend_bos,
        #**sae_cfg.model_kwargs,
    )[1].cache_dict[sae_cfg.hook_name]


def get_sae_activations(model_activations: torch.tensor, sae: SAE, sparsity_threshold: float = 1e-6,
                        num_activations_required: int | None = None):
    activations = sae.encode(model_activations)
    activations = einops.rearrange(activations, 'b s a -> (b s) a')
    activations = activations.cpu()

    if num_activations_required is not None:
        activations = activations[:num_activations_required]

    activations = torch.where(
        torch.abs(activations) < sparsity_threshold,
        torch.zeros_like(activations),
        activations
    )

    return activations.to_sparse()


def get_activations(sae: SAE, model: HookedTransformer, dataset: IterableDataset, batch_size: int = 32,
                    total_tokens: int = 1_000_000, sparsity_threshold: float = 1e-6):
    activations = []
    num_tokens = 0
    progress = tqdm(total=total_tokens, desc="Extracting activations")
    for batch in dataset.iter(batch_size):
        batch_tokens = torch.tensor(
            batch['input_ids'],
            dtype=torch.long,
            device=device,
            requires_grad=False
        )[:, :sae.cfg.context_size]
        model_activations = get_model_activations(model, batch_tokens, sae.cfg)
        required_tokens = min(total_tokens - num_tokens, batch_size * sae.cfg.context_size)
        sae_activations = get_sae_activations(model_activations, sae, sparsity_threshold, required_tokens)

        activations.append(sae_activations)

        num_tokens += sae_activations.shape[0]
        progress.update(sae_activations.shape[0])

        if num_tokens >= total_tokens:
            break

    return torch.cat(activations)


def get_dead_feature_indices(activations):
    dead_features_mask = get_dead_features_mask(activations)
    return torch.arange(activations.shape[1])[dead_features_mask]


def get_dead_features_mask(activations):
    alive_feature_indices = activations.coalesce().indices()[1].unique()
    dead_features_mask = torch.ones(activations.shape[1], dtype=torch.bool)
    dead_features_mask[alive_feature_indices] = False
    return dead_features_mask


def cosine_distance(activations_orig, activations_matched):
    activations_orig_norm = (activations_orig ** 2).sum(dim=0, keepdim=True).sqrt().to_dense()
    activations_matched_norm = (activations_matched ** 2).sum(dim=0, keepdim=True).sqrt().to_dense()
    activations_orig = activations_orig * (1 / activations_orig_norm)
    activations_matched = activations_matched * (1 / activations_matched_norm)

    similarity = activations_orig.T @ activations_matched

    # Now convert to dense numpy array
    pairwise_distances = 1 - similarity.to_dense().numpy()

    return pairwise_distances


def get_pairwise_distances(activations_orig, activations_matched, metric='cosine'):
    if metric == 'cosine':
        pairwise_distances = cosine_distance(activations_orig, activations_matched)
    else:
        raise ValueError(f"Invalid metric: {metric}")
    return pairwise_distances


def match_features(activations_orig, activations_matched, metric='cosine'):
    """
    Match features given two corresponding sets of activations by solving linear sum assignment.
    """
    pairwise_distances = get_pairwise_distances(activations_orig, activations_matched, metric=metric)

    # Solve linear assignment
    indices_orig, indices_matched = linear_sum_assignment(pairwise_distances)

    # Remove dead features
    dead_features_mask_orig = get_dead_features_mask(activations_orig)
    dead_features_mask_matched = get_dead_features_mask(activations_matched)

    dead_features_mask = dead_features_mask_orig | dead_features_mask_matched
    dead_feature_indices = torch.argwhere(dead_features_mask).squeeze()

    """
    dead_features_mask_matched = dead_features_mask_matched[indices_matched]

    mask_orig = ~dead_features_mask_orig[indices_orig]
    indices_orig = indices_orig[mask_orig]
    indices_matched = indices_matched[mask_orig]

    mask_matched = ~dead_features_mask_matched[indices_matched]
    indices_orig = indices_orig[mask_matched]
    indices_matched = indices_matched[mask_matched]
    """

    matched_distances = pairwise_distances[indices_orig, indices_matched]
    matched_distances[dead_feature_indices] = -1

    return indices_orig, indices_matched, matched_distances, dead_feature_indices