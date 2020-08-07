from typing import List, Mapping, Optional

import torch
import torch.nn.functional as F

Outputs = Mapping[str, List[torch.Tensor]]


def cross_entropy_with_probs(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Calculate cross-entropy loss when targets are probabilities (floats), not ints.

    Combines the code from snorkel.classification.loss.cross_entropy_with_probs function (L.67-82)
    simulating cross entropy functionality in pytorch from
    https://github.com/snorkel-team/snorkel/blob/master/snorkel/classification/loss.py
    with parts of pytorch.nn.functional.nll_loss (L.55-62) to handle tensors with ndim > 2 from
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#nll_loss

    PyTorch's F.cross_entropy() method requires integer labels; it does not accept
    probabilistic labels. We can, however, simulate such functionality with a for loop,
    calculating the loss contributed by each class and accumulating the results.
    Libraries such as keras do not require this workaround, as methods like
    "categorical_crossentropy" accept float labels natively.

    Note that the method signature is intentionally very similar to F.cross_entropy()
    so that it can be used as a drop-in replacement when target labels are changed from
    from a 1D tensor of ints to a 2D tensor of probabilities.

    Parameters
    ----------
    input
        A tensor of logits
    target
        A tensor of probabilistic target labels
    weight
        An optional [num_classes] array of weights to multiply the loss by per class
    reduction
        One of "none", "mean", "sum", indicating whether to return one loss per data
        point, the mean loss, or the sum of losses

    Returns
    -------
    torch.Tensor
        The calculated loss

    Raises
    ------
    ValueError
        If an invalid reduction keyword is submitted
    """
    dim = input.dim()
    if dim < 2:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))
    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))
    n = input.size(0)   # number of points
    c = input.size(1)   # number of classes
    out_size = (n,) + input.size()[2:]
    if target.size()[1:-1] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))
    cum_losses = input.new_zeros(out_size)
    for y in range(c):
        target_temp = input.new_full(out_size, y, dtype=torch.long)
        y_loss = F.cross_entropy(input, target_temp, reduction="none")
        if weight is not None:
            y_loss = y_loss * weight[y]
        cum_losses += target[..., y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")
