import numpy as np

from hft.ml.rl_train import Policy, grid, pbo


def test_grid_is_the_sixteen_preregistered_configurations():
    configs = grid()
    assert len(configs) == 16 and len({c["id"] for c in configs}) == 16
    assert {c["track"] for c in configs} == {"A", "B"}


def test_pbo_is_low_for_a_consistent_winner_and_high_for_noise():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, (480, 8))
    winner = noise.copy()
    winner[:, 3] += 0.01
    assert pbo(winner) < 0.05
    assert 0.2 < pbo(noise) < 0.8


def test_masked_actions_are_never_chosen():
    import torch

    policy = Policy(15, 5, "gru")
    mask = torch.tensor([[True, False], [False, True]])
    logits, value = policy(torch.zeros(2, 60, 15), torch.zeros(2, 5), mask)
    assert logits.argmax(-1).tolist() == [0, 1] and value.shape == (2,)
