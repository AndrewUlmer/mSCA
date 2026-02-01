import torch
import matplotlib.pyplot as plt

sparsity_range = [
    0.0,
    0.001,
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.035,
    0.04,
    0.045,
    0.05,
    0.055,
    0.06,
    0.065,
    0.07,
    0.075,
    0.08,
    0.085,
    0.09,
    0.095,
    0.1,
    # 0.2,
    # 0.3,
    # 0.4,
    # 0.5,
]
pre = [
    torch.load(
        f"./experiments/simulation/sparsity_sweep_decoder_single_trial/pre_{sparsity:.4f}.pt"
    )
    for sparsity in sparsity_range
]
post = [
    torch.load(
        f"./experiments/simulation/sparsity_sweep_decoder_single_trial/post_{sparsity:.4f}.pt"
    )
    for sparsity in sparsity_range
]

print("something")
