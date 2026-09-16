import pytest
import torch

from glitches.steps import step_from_sigmas

SCHEDULE = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])


@pytest.mark.parametrize(
    ("sigma", "step"),
    [(1.0, 0), (0.75, 1), (0.5, 2), (0.25, 3), (0.9, 0), (0.6, 1), (0.3, 2), (0.1, 3), (1.5, 0), (0.0, 3)],
)
def test_schedule_points_and_in_between_sigmas(sigma, step):
    assert step_from_sigmas(torch.tensor([sigma, sigma]), SCHEDULE) == step


def test_float_error_just_above_a_schedule_point_counts_as_that_step():
    assert step_from_sigmas(torch.tensor([0.5 + 1e-8], dtype=torch.float64), SCHEDULE.double()) == 2


def test_uses_largest_sigma_in_the_batch():
    assert step_from_sigmas(torch.tensor([0.5, 0.75]), SCHEDULE) == 1


def test_shortened_schedule_counts_from_its_first_entry():
    assert step_from_sigmas(torch.tensor([0.5]), torch.tensor([0.5, 0.25, 0.0])) == 0


def test_single_step_schedule():
    assert step_from_sigmas(torch.tensor([1.0]), torch.tensor([1.0, 0.0])) == 0


def test_schedule_too_short():
    with pytest.raises(ValueError, match="at least 2 values"):
        step_from_sigmas(torch.tensor([1.0]), torch.tensor([1.0]))
