import torch
import matplotlib.pyplot as plt

sparsity_range = [
    0.000,
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.006,
    0.007,
    0.008,
    0.009,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    # 0.035,
    # 0.040,
    # 0.045,
    # 0.05,
    # 0.055,
    # 0.06,
    # 0.065,
    # 0.07,
    # 0.075,
    # 0.08,
    # 0.085,
    # 0.09,
    # 0.095,
    # 0.1,
]
pre = [
    torch.load(
        f"./experiments/reaching/sparsity_sweep_orth_linear/pre_{sparsity:.4f}.pt"
    )
    for sparsity in sparsity_range
]
post = [
    torch.load(
        f"./experiments/reaching/sparsity_sweep_orth_linear/post_{sparsity:.4f}.pt"
    )
    for sparsity in sparsity_range
]

print("something")
