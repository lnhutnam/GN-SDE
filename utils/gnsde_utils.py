import torch


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(
        b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign()
    )
    return a / b
