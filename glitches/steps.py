"""Map ComfyUI's per-call sigma to a sampling step index (torch only)."""

from __future__ import annotations

import torch


def step_from_sigmas(sigmas: torch.Tensor, sample_sigmas: torch.Tensor) -> int:
    """Index of the schedule interval containing the current sigma.

    ``sample_sigmas`` is the descending schedule of N+1 values that the sampler runs; a call at
    ``sample_sigmas[i]`` is step ``i`` and calls between two entries belong to the earlier one.
    A custom schedule with repeated adjacent sigma values assigns calls to the later of the
    equal entries (ComfyUI's built-in schedulers are strictly decreasing until the final 0, so
    this only matters for custom schedules).
    """
    schedule = [float(value) for value in torch.as_tensor(sample_sigmas).detach().flatten().cpu()]
    if len(schedule) < 2:
        raise ValueError("sample_sigmas must contain at least 2 values")
    sigma = float(torch.as_tensor(sigmas).detach().max().cpu())
    n_steps = len(schedule) - 1
    eps = 1e-6 * max(schedule[0], 1.0)
    step = sum(1 for value in schedule[1:n_steps] if value >= sigma - eps)
    return min(max(step, 0), n_steps - 1)
