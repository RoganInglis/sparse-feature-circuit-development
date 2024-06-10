import torch
from match_sae_features import get_dataset, get_full_cfg, device


def test_dataset():
    dataset_1 = get_dataset(get_full_cfg('/home/rogan/git_repos/sparse-feature-circuit-development/scripts/checkpoints/0y98mkfh/final_400003072'))
    dataset_2 = get_dataset(get_full_cfg('/home/rogan/git_repos/sparse-feature-circuit-development/scripts/checkpoints/4ud9soq9/final_400003072'))

    iterations = 0
    for batch_1, batch_2 in zip(dataset_1.iter(32), dataset_2.iter(32)):
        batch_1_tensor = torch.tensor(batch_1['input_ids'], dtype=torch.long, device=device)
        batch_2_tensor = torch.tensor(batch_2['input_ids'], dtype=torch.long, device=device)
        assert torch.allclose(batch_1_tensor, batch_2_tensor)
        iterations += 1
        if iterations > 10:
            break
