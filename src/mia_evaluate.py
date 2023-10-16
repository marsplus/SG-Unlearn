import torch
import numpy as np
import torch.nn as nn
from sklearn.svm import LinearSVC
from sklearn import linear_model, model_selection


def simple_mia(model,
               forget_loader, 
               test_loader, 
               n_splits=10, 
               random_state=0,
               device='cpu'):
    forget_losses = compute_losses(model, forget_loader, device)
    test_losses = compute_losses(model, test_loader, device)
    ## ensure a balanced dataset
    ns = min(len(forget_losses), len(test_losses))
    np.random.shuffle(forget_losses)
    forget_losses, test_losses = forget_losses[:ns], test_losses[:ns]
    sample_loss = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    members = [0] * len(test_losses) + [1] * len(forget_losses)

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    acc = model_selection.cross_val_score(attack_model, sample_loss, members, cv=cv, scoring="accuracy").mean()
    auc = model_selection.cross_val_score(attack_model, sample_loss, members, cv=cv, scoring="roc_auc").mean()
    f1 = model_selection.cross_val_score(attack_model, sample_loss, members, cv=cv, scoring="f1").mean()
    return (acc, auc, f1, forget_losses, test_losses)


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