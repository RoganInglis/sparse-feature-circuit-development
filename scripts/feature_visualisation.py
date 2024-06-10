from typing import List

from datasets import load_dataset
from transformer_lens import HookedTransformer
import json
from sae_lens import SAE

# Standard imports
import os
import torch
from tqdm import tqdm
import plotly.express as px

# Imports for displaying vis in Colab / notebook
import webbrowser
from sae_vis.data_config_classes import SaeVisConfig
from sae_vis.data_storing_fns import SaeVisData
from random import shuffle

torch.set_grad_enabled(False)

# For the most part I'll try to import functions and classes near where they are used
# to make it clear where they come from.

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


def display_vis_inline(filename: str):
    '''
    Displays the HTML files in Colab. Uses global `PORT` variable defined in prev cell, so that each
    vis has a unique port without having to define a port within the function.
    '''
    webbrowser.open(filename)


def get_model(sae, full_cfg):
    model_from_pretrained_kwargs = full_cfg.get('model_from_pretrained_kwargs') or {}

    return HookedTransformer.from_pretrained(sae.cfg.model_name, device=device, **model_from_pretrained_kwargs)


def get_dataset(full_cfg):
    dataset = load_dataset(
        path=full_cfg['dataset_path'],
        split="train",
        streaming=True,
    )
    return iter(dataset)


def get_tokens_for_vis(dataset, sae):
    tokens = []
    while len(tokens) < 10000:
        tokens.append(
            torch.tensor(
                next(dataset)['input_ids'],
                dtype=torch.long,
                device=device,
                requires_grad=False,
            )[None, :sae.cfg.context_size]
        )
    return torch.cat(tokens, dim=0)


def visualise_features_for_sae_path(sae_path: str, feature_indices: List[int],
                                    mapped_feature_indices: List[int] | None = None, out_dir: str = '.',
                                    auto_open: bool = False):
    os.makedirs(out_dir, exist_ok=True)

    sae = SAE.load_from_pretrained(
        path=sae_path,
        device=device
    )

    with open(os.path.join(sae_path, "cfg.json")) as f:
        full_cfg = json.load(f)

    if full_cfg['scale_sparsity_penalty_by_decoder_norm']:
        sae.fold_W_dec_norm()

    model = get_model(sae, full_cfg)
    dataset = get_dataset(full_cfg)

    feature_vis_config_gpt = SaeVisConfig(
        hook_point=sae.cfg.hook_name,
        features=feature_indices,
        # batch_size=3*2048,
        minibatch_size_tokens=32,
        verbose=True,
    )

    tokens = get_tokens_for_vis(dataset, sae)

    sae_vis_data_gpt = SaeVisData.create(
        encoder=sae,
        model=model,  # type: ignore
        tokens=tokens,  # type: ignore
        cfg=feature_vis_config_gpt,
    )

    for i, feature in enumerate(feature_indices):
        if mapped_feature_indices is not None:
            feature_name = mapped_feature_indices[i]
        else:
            feature_name = feature
        filename = os.path.join(out_dir, f"{feature_name}_ckpt_{full_cfg.get('model_from_pretrained_kwargs', {}).get('checkpoint_index')}_feature_vis_demo_gpt.html")
        sae_vis_data_gpt.save_feature_centric_vis(filename, feature)
        if auto_open:
            display_vis_inline(filename)


def main():
    # the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
    # Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
    # We also return the feature sparsities which are stored in HF for convenience.

    sae_path = '/home/rogan/git_repos/sparse-feature-circuit-development/scripts/checkpoints/p0hcxgno/final_400003072'
    sae = SAE.load_from_pretrained(
        path=sae_path,
        device=device
    )

    with open(os.path.join(sae_path, "cfg.json")) as f:
        full_cfg = json.load(f)

    if full_cfg['scale_sparsity_penalty_by_decoder_norm']:
        sae.fold_W_dec_norm()

    model = get_model(sae, full_cfg)
    dataset = get_dataset(full_cfg)

    test_feature_idx_gpt = list(range(20))

    feature_vis_config_gpt = SaeVisConfig(
        hook_point=sae.cfg.hook_name,
        features=test_feature_idx_gpt,
        #batch_size=3*2048,
        minibatch_size_tokens=32,
        verbose=True,
    )

    tokens = get_tokens_for_vis(dataset, sae)

    sae_vis_data_gpt = SaeVisData.create(
        encoder=sae,
        model=model, # type: ignore
        tokens=tokens,  # type: ignore
        cfg=feature_vis_config_gpt,
    )

    for feature in test_feature_idx_gpt:
        filename = f"{feature}_feature_vis_demo_gpt.html"
        sae_vis_data_gpt.save_feature_centric_vis(filename, feature)
        display_vis_inline(filename)


if __name__ == '__main__':
    main()
