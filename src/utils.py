import math
import warnings
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn import linear_model, model_selection
from sklearn.datasets import make_blobs
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset


def wasserstein_distance_1d(dist1, dist2):
    dist1_sorted = sorted(dist1)
    dist2_sorted = sorted(dist2)
    return np.mean(np.abs(np.array(dist1_sorted) - np.array(dist2_sorted)))


def random_split(dataset, lengths, generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def print_parms(model):
    for name, parameter in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter shape: {parameter.shape}")
        print(f"Parameter values: {parameter}")
        print()


def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    acc = model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    ).mean()
    auc = model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="roc_auc"
    ).mean()
    return (acc, auc)


def compute_losses(net, loader, device):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        logits = net(inputs)
        losses = criterion(logits, targets).detach().cpu().numpy()
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)


def evaluate_accuracy(model, data_loader, device="cpu"):
    """
    Evaluate the accuracy of a PyTorch model using a DataLoader.

    Parameters:
    - model: A PyTorch model.
    - data_loader: A PyTorch DataLoader with the dataset.

    Returns:
    - Accuracy of the model on the dataset.
    """

    # Ensure the model is in evaluation mode
    model.eval()

    # Variables to store total correct predictions and total predictions
    total_correct = 0
    total_predictions = 0

    # No need to track gradients during evaluation
    with torch.no_grad():
        try:
            for inputs, labels in data_loader:
                # Move data to the same device as the model if necessary
                inputs, labels = inputs.to(device), labels.to(device)

                # Compute model's predictions
                outputs = model(inputs)

                # Get the predicted class for each example in the batch
                _, predicted = torch.max(outputs, 1)

                # Update total predictions and total correct predictions
                total_predictions += labels.size(0)
                total_correct += (predicted == labels).sum().item()
        except ValueError as e:
            for inputs, masks, labels in data_loader:
                # Move data to the same device as the model if necessary
                inputs, masks, labels = (
                    inputs.to(device),
                    masks.to(device),
                    labels.to(device),
                )

                # Compute model's predictions
                outputs = model(inputs, masks)

                # Get the predicted class for each example in the batch
                _, predicted = torch.max(outputs, 1)

                # Update total predictions and total correct predictions
                total_predictions += labels.size(0)
                total_correct += (predicted == labels).sum().item()

    # Compute accuracy
    accuracy = total_correct / total_predictions
    return accuracy


# Define a custom dataset class
class BinaryClassificationDataset(Dataset):
    def __init__(self, num_samples, num_features):
        super(BinaryClassificationDataset, self).__init__()
        self.num_samples = num_samples
        self.num_features = num_features
        self.data, self.labels = self.generate_data()

    def generate_data(self, std: float = 3.0):
        ## generate two Gaussian distributed clusters
        X, y = make_blobs(
            self.num_samples,
            self.num_features,
            centers=np.array(
                [2 * np.ones(self.num_features), -2 * np.ones(self.num_features)]
            ),
            cluster_std=std,
        )
        ## to avoid latter float64 v.s. float32 issues
        _func = lambda x: (
            torch.from_numpy(x).float()
            if x.dtype != int
            else torch.from_numpy(x).long()
        )
        return map(_func, [X, y])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
