import logging
import os
import pdb
import pickle
import sys
import time

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cvxpylayers.torch import CvxpyLayer
from evaluate import evaluate_model

# from memory_profiler import profile
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from utils import BinaryClassificationDataset, wasserstein_distance_1d

# from qpth.qp import QPFunction


# self.device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ATTACKER = 0
REF_VERSION = "1.13.1"
from torch.utils.model_zoo import load_url as load_state_dict_from_url


# An MLP class for debugging purposes
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)


class DefenderOPT(nn.Module):
    def __init__(
        self,
        D_r: DataLoader = None,
        D_f: DataLoader = None,
        D_v: DataLoader = None,
        D_t: DataLoader = None,
        dim: int = 1,
        cv: int = 5,
        batch_size: int = 128,
        num_epoch: int = 10,
        baseline_mode: bool = False,
        regular_type: str = "l2",
        regular_strength: float = 0.5,
        num_class: int = 2,
        attacker_lr: float = 0.01,
        defender_lr: float = 0.01,
        with_attacker: bool = True,
        attacker_reg: float = 0.0,
        wasserstein_coeff: float = 0.5,
        output_dir: str = None,
        seed: int = 42,
        weight_decay: float = 5e-4,
        momentum: float = 0.9,
        device_id: int = 0,
        fine_tune: bool = True,
        att_classifier: str = "SVM",
        attacker_strength: float = 1.0,
        save_checkpoint: bool = True,
        classwise: bool = False,
    ):
        """
        dim: the dimension of the score vectors.
             If dim=1, the scores are the final scalar losses (e.g., cross-entropy).
             Otherwise, the scores are the concatenation of the scalar losses and the logits.
        baseline_mode: if true, no unlearning is performend; only evaluating an input net.
        regular_type: the regularization added to the computation of 1d wasserstein distance; either `l2` or `kl`
        regular_strength: the strength of the regularization. The larger the strength, the smoother the smoother the sorting.
        attacker_reg: the amount of regularization applied to the logistic regression of the MIA attack.
        wasserstein_coeff: the amount of 1d wasserstein in the attacker's utility
        """
        super(DefenderOPT, self).__init__()
        self.retain_loader = D_r
        self.forget_loader = D_f
        self.val_loader = D_v
        self.test_loader = D_t
        self.num_class = num_class
        self.dim = dim
        self.cv = cv
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.baseline_mode = baseline_mode
        self.regular_type = regular_type
        self.regular_strength = regular_strength
        self.attacker_lr = attacker_lr
        self.defender_lr = defender_lr
        self.with_attacker = with_attacker
        self.attacker_reg = attacker_reg
        self.wasserstein_coeff = wasserstein_coeff
        self.output_dir = output_dir
        self.seed = seed
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.att_classifier = att_classifier
        self.attacker_strength = attacker_strength
        self.save_checkpoint = save_checkpoint
        self.classwise = classwise

        ## cross-validation split used in the attacker's optimization
        ## it's initialized here for having consistent number of train/test data
        self.kf = StratifiedShuffleSplit(
            n_splits=self.cv, test_size=0.3, random_state=self.seed
        )

        if self.attacker_strength == 0.0:
            self.with_attacker = 0

    def _save_ckpt(self, model, epoch):
        SG_data = {
            "retain": self.retain_loader.dataset,
            "val": self.val_loader.dataset,
            "forget": self.forget_loader.dataset,
            "test": self.test_loader.dataset,
        }
        ## save evaluation results
        eval_ret = evaluate_model(model, SG_data, seed=self.seed, device=self.device)
        torch.save(
            eval_ret,
            os.path.join(
                self.output_dir,
                f"eval_num_epoch_{epoch}_cv_{self.cv}_dim_{self.dim}_atts_{self.attacker_strength}_seed_{self.seed}.pth",
            ),
        )
        ## save model checkpoints
        checkpoint = model.state_dict()
        torch.save(
            checkpoint,
            os.path.join(
                self.output_dir,
                f"SGcheckpoint_num_epoch_{epoch}_cv_{self.cv}_dim_{self.dim}_atts_{self.attacker_strength}_seed_{self.seed}.pth",
            ),
        )

    def unlearn(self, net):
        """
        The unlearning algorithm, a.k.a, the defender's optimization problem.

        Parameters:
            net: the unlearned model

        Returns:
            net: the unlearned model
        """
        ## the usual performance on the retain data
        # train_loader = DataLoader(self.retain_loader.dataset, batch_size=self.batch_size, shuffle=True)
        # test_loader = DataLoader(self.test_loader.dataset, batch_size=self.batch_size, shuffle=False)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            net.parameters(),
            lr=self.defender_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epoch
        )

        ## check the performance before any optimization
        if self.baseline_mode:
            test_accuracy = self._evaluate_accuracy(
                net, self.test_loader, device=self.device
            )
            forget_accuracy = self._evaluate_accuracy(
                net, self.forget_loader, device=self.device
            )
            retain_accuracy = self._evaluate_accuracy(
                net, self.retain_loader, device=self.device
            )
            MIA_accuracy, MIA_recall, MIA_auc = DefenderOPT._evaluate_MIA(
                net,
                self.forget_loader,
                self.val_loader,
                dim=self.dim,
                device=self.device,
                seed=self.seed,
            )
            print(
                f"test accuracy: {test_accuracy:.4f}, "
                f"forget accuracy: {forget_accuracy:.4f}, "
                f"retain accuracy: {retain_accuracy:.4f}, "
                f"MIA accuracy: {MIA_accuracy.item():.4f}, "
                f"MIA auc: {MIA_auc.item():.4f}, "
                f"MIA recall: {MIA_recall.item():.4f}"
            )
            epoch = 0
            self._save_ckpt(net, epoch)
            return net

        net.train()
        for epoch in range(self.num_epoch):
            t_start = time.time()
            try:
                for inputs, targets in self.retain_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    u_d = -loss_func(outputs, targets)
                    (-u_d).backward()
                    optimizer.step()
            except ValueError as e:
                for inputs, masks, targets in self.retain_loader:
                    inputs, masks, targets = (
                        inputs.to(self.device),
                        masks.to(self.device),
                        targets.to(self.device),
                    )
                    optimizer.zero_grad()
                    outputs = net(inputs, masks)
                    u_d = -loss_func(outputs, targets)
                    (-u_d).backward()
                    optimizer.step()

            t_att_start = time.time()
            if self.with_attacker:
                ## the attacker's utility
                ## _set_lr enables different lr for the defender and attacker
                self._set_lr(optimizer, new_lr=self.attacker_lr)
                optimizer.zero_grad()
                ## the gradient of the attacker's optimization w.r.t. the model parameters:
                ## Let N = 2*min(#forget data, #test data); B = batch size = 2*(#forget data, #test data)
                ## batched version (memory intensive): (1/N) * (dL / dw)
                ## mini batch version (memory friendly): (1/B) * (dL / dw), so we need to multiply by a factor of (B/N)
                N = min(len(self.forget_loader.dataset), len(self.val_loader.dataset))
                for forget_batch, val_batch in zip(self.forget_loader, self.val_loader):
                    # Determine if masks are included
                    if len(forget_batch) == 2:
                        forget_data, forget_targets = forget_batch
                        val_data, val_targets = val_batch
                        args = (forget_data, forget_targets, val_data, val_targets, net)
                    else:
                        forget_data, forget_masks, forget_targets = forget_batch
                        val_data, val_masks, val_targets = val_batch
                        args = (
                            forget_data,
                            forget_targets,
                            val_data,
                            val_targets,
                            net,
                            forget_masks,
                            val_masks,
                        )
                    B = min(forget_data.shape[0], val_data.shape[0])
                    # Compute attacker likelihood, accuracy, and other metrics
                    att_lik, att_acc, _ = self._attacker_opt(*args)
                    # Calculate the attacker's utility
                    u_a = att_lik * self.attacker_strength * (B / N)
                    u_a.backward()
                optimizer.step()
            scheduler.step()
            ## revert the lr back
            self._set_lr(optimizer, new_lr=self.defender_lr)
            ## time in minutes (excluding evaluation, etc.)
            t_att = (time.time() - t_att_start) / 60.0
            t_all = (time.time() - t_start) / 60.0

            ## check the normal performance on validation set
            test_accuracy = self._evaluate_accuracy(
                net, self.test_loader, device=self.device
            )
            retain_accuracy = self._evaluate_accuracy(
                net, self.retain_loader, device=self.device
            )
            forget_accuracy = self._evaluate_accuracy(
                net, self.forget_loader, device=self.device
            )
            MIA_accuracy, MIA_recall, MIA_auc = DefenderOPT._evaluate_MIA(
                net,
                self.forget_loader,
                self.val_loader,
                dim=self.dim,
                device=self.device,
                seed=self.seed,
            )
            print(
                f"Epoch [{epoch+1}/{self.num_epoch}], ",
                f"U_d: {u_d.item():.4f}, ",
                f"U_a: {u_a.item():.4f}, " if self.with_attacker else f"U_a: None, ",
                f"test accuracy: {test_accuracy:.4f}, ",
                f"forget accuracy: {forget_accuracy:.4f}, ",
                f"retain accuracy: {retain_accuracy:.4f}, ",
                (
                    f"att accuracy: {att_acc.item():.4f}, "
                    if self.with_attacker
                    else f"att accuracy: None, "
                ),
                f"MIA accuracy: {MIA_accuracy.item():.4f}, ",
                f"MIA auc: {MIA_auc.item():.4f}, ",
                f"MIA recall: {MIA_recall.item():.4f}, ",
            )
            print(
                f"time/epoch: {t_all:.4f} min; attacker opt: {t_att:.4f} ({t_att/t_all:.2f})"
            )

            if self.save_checkpoint and (epoch + 1) % 5 == 0:
                self._save_ckpt(net, epoch + 1)

    @staticmethod
    def _generate_scores(
        net,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mode="train",
        dim: int = 1,
        device="cpu",
    ):
        """
        Feed the test and forget sets through the unlearned model,
        and collect the output scores with the corresponding class labels.

        Parameters:
            inputs: the input data (either forget or test)

        Return:
            scores: the score vectors with shape (n, dim)
            clas: class labels with shape (n, ).
        """
        net.train() if mode == "train" else net.eval()
        criterion = nn.CrossEntropyLoss(reduction="none")
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        losses = criterion(outputs, targets)
        new_score = (
            losses[:, None]
            if dim == 1
            else torch.cat((outputs, losses[:, None]), axis=1)
        )
        return new_score

    @staticmethod
    def _generate_scores_w_masks(
        net,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        targets: torch.Tensor,
        mode="train",
        dim: int = 1,
        device="cpu",
    ):
        """
        This function is just to handle the input mask used in NLP tasks
        """
        net.train() if mode == "train" else net.eval()
        criterion = nn.CrossEntropyLoss(reduction="none")
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
        outputs = net(inputs, masks)
        losses = criterion(outputs, targets)
        new_score = (
            losses[:, None]
            if dim == 1
            else torch.cat((outputs, losses[:, None]), axis=1)
        )
        return new_score

    @staticmethod
    def _set_lr(optimizer, new_lr):
        """
        set the learn rate of the input optimizer
        """
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    @staticmethod
    @torch.no_grad()
    def _evaluate_accuracy(net, eval_loader: DataLoader = None, device="cpu"):
        """
        A template code to evaluate a model's accuracy on a dataset.
        """
        net.eval()
        total_correct = 0
        total_samples = 0
        for data in eval_loader:
            if len(data) == 2:
                inputs, targets = data
                outputs = net(inputs.to(device))
            else:
                inputs, masks, targets = data
                outputs = net(inputs.to(device), masks.to(device))
            targets = targets.to(device)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
        acc = total_correct / total_samples
        return acc

    @staticmethod
    @torch.no_grad()
    def _evaluate_MIA(
        net,
        forget_loader: DataLoader = None,
        test_loader: DataLoader = None,
        dim: int = 1,
        device: str = "cpu",
        seed: int = 42,
    ) -> np.array:
        """
        This function mimics the `simple_mia` function in the starting kit.
        Returns:
            MIA_accuracy (np.array): the averaged MIA accuracy.

        """
        all_scores = []
        all_members = []
        try:
            for (forget_data, forget_targets), (test_data, test_targets) in zip(
                forget_loader, test_loader
            ):
                ns = min(forget_data.shape[0], test_data.shape[0])
                forget_scores = DefenderOPT._generate_scores(
                    net,
                    forget_data,
                    forget_targets,
                    mode="eval",
                    dim=dim,
                    device=device,
                )
                test_scores = DefenderOPT._generate_scores(
                    net, test_data, test_targets, mode="eval", dim=dim, device=device
                )  # the naming is a bit bad here :(
                forget_members, test_members = (
                    torch.ones(ns, device=device),
                    torch.zeros(ns, device=device),
                )
                all_scores.append(
                    torch.cat((forget_scores[:ns], test_scores[:ns]), axis=0)
                )  # shape=(ns, dim)
                all_members.append(
                    torch.cat((forget_members, test_members), axis=0)
                )  # shape=(ns, )
        except ValueError as e:
            for (forget_data, forget_masks, forget_targets), (
                test_data,
                test_masks,
                test_targets,
            ) in zip(forget_loader, test_loader):
                ns = min(forget_data.shape[0], test_data.shape[0])
                forget_scores = DefenderOPT._generate_scores_w_masks(
                    net,
                    forget_data,
                    forget_masks,
                    forget_targets,
                    mode="eval",
                    dim=dim,
                    device=device,
                )
                test_scores = DefenderOPT._generate_scores_w_masks(
                    net,
                    test_data,
                    test_masks,
                    test_targets,
                    mode="eval",
                    dim=dim,
                    device=device,
                )  # the naming is a bit bad here :(
                forget_members, test_members = (
                    torch.ones(ns, device=device),
                    torch.zeros(ns, device=device),
                )
                all_scores.append(
                    torch.cat((forget_scores[:ns], test_scores[:ns]), axis=0)
                )  # shape=(ns, dim)
                all_members.append(
                    torch.cat((forget_members, test_members), axis=0)
                )  # shape=(ns, )
        all_scores = torch.cat(all_scores, axis=0)
        all_members = torch.cat(all_members, axis=0)

        if int(torch.__version__[0]) > int(REF_VERSION[0]) or int(
            torch.__version__.split(".")[1]
        ) >= int(REF_VERSION.split(".")[1]):
            all_scores = all_scores.numpy(force=True)
            all_members = all_members.numpy(force=True)
        else:
            all_scores = all_scores.detach().cpu().numpy()
            all_members = all_members.detach().cpu().numpy()

        unique_members = np.unique(all_members)
        if not np.all(unique_members == np.array([0, 1])):
            raise ValueError("members should only have 0 and 1s")

        # attack_model = LogisticRegression()
        attack_model = LinearSVC()
        cv = StratifiedShuffleSplit(n_splits=10, random_state=seed)
        MIA_accuracy = cross_val_score(
            attack_model, all_scores, all_members, cv=cv, scoring="accuracy"
        ).mean()
        MIA_recall = cross_val_score(
            attack_model, all_scores, all_members, cv=cv, scoring="recall"
        ).mean()
        MIA_auc = cross_val_score(
            attack_model, all_scores, all_members, cv=cv, scoring="roc_auc"
        ).mean()
        return (MIA_accuracy, MIA_recall, MIA_auc)

    def _classwise_processing(self, data, targets, forget_classes):
        ## remove the data from the `forget class`
        data_copy = data.clone()
        targets_copy = targets.clone()
        mask = ~torch.any(targets.unsqueeze(1) == forget_classes.unsqueeze(0), dim=1)
        data_copy = data_copy[mask]
        targets_copy = targets[mask]
        return (data_copy, targets_copy)

    def _attacker_opt(
        self,
        forget_data: torch.Tensor,
        forget_targets: torch.Tensor,
        test_data: torch.Tensor,
        test_targets: torch.Tensor,
        net=None,
        forget_masks=None,
        test_masks=None,
    ):
        """
        The attacker's utility maximizing problem.


        Returns:
            total_ua: the attacker's utility summed across the folds.

        """
        assert net is not None, "The input net is None.\n"
        ## handle classwise setting
        if (
            self.classwise
            and np.intersect1d(
                forget_targets.squeeze().numpy(), test_targets.squeeze().numpy()
            ).size
            != 0
        ):
            # test_data, test_targets = self._classwise_processing(test_data, test_targets, torch.unique(forget_targets))
            test_data, test_targets = self._classwise_processing(
                test_data, test_targets, torch.tensor([0], device=self.device)
            )

        ## make sure the auditing set is balanced
        ns = min(forget_data.shape[0], test_data.shape[0])
        if forget_masks is None and test_masks is None:
            forget_scores = DefenderOPT._generate_scores(
                net,
                forget_data,
                forget_targets,
                mode="train",
                dim=self.dim,
                device=self.device,
            )
            ## the naming below is bad; the `test_data` actually represents the `val_data`
            test_scores = DefenderOPT._generate_scores(
                net,
                test_data,
                test_targets,
                mode="train",
                dim=self.dim,
                device=self.device,
            )
        else:
            forget_scores = DefenderOPT._generate_scores_w_masks(
                net,
                forget_data,
                forget_masks,
                forget_targets,
                mode="train",
                dim=self.dim,
                device=self.device,
            )
            ## the naming below is bad; the `test_data` actually represents the `val_data`
            test_scores = DefenderOPT._generate_scores_w_masks(
                net,
                test_data,
                test_masks,
                test_targets,
                mode="train",
                dim=self.dim,
                device=self.device,
            )
        all_scores = torch.cat(
            (forget_scores[:ns], test_scores[:ns]), axis=0
        )  # shape=(ns, dim)
        all_clas = torch.cat((forget_targets[:ns], test_targets[:ns]), axis=0).to(
            torch.long
        )  # shape=(ns, )
        forget_members, test_members = (
            torch.ones(ns, device=self.device),
            torch.zeros(ns, device=self.device),
        )

        ## compute 1d wasserstein distance (DEPRECATED)
        wasserstein_dist = torch.tensor(0.0, device=self.device)

        all_members = torch.cat((forget_members, test_members), axis=0).to(
            torch.long
        )  # shape=(ns, )

        # shuffle the data
        idx = torch.randperm(all_scores.shape[0])
        all_scores = all_scores[idx]
        all_members = all_members[idx]
        all_clas = all_clas[idx]
        (
            all_scores.to(self.device),
            all_members.to(self.device),
            all_clas.to(self.device),
        )

        ## below is just to handle platform issues
        if int(torch.__version__[0]) > int(REF_VERSION[0]) or int(
            torch.__version__.split(".")[1]
        ) >= int(REF_VERSION.split(".")[1]):
            all_scores_numpy = all_scores.numpy(force=True)[:, np.newaxis]
            all_clas_numpy = all_clas.numpy(force=True)[:, np.newaxis]
        else:
            all_scores_numpy = all_scores.detach().cpu().numpy()[:, np.newaxis]
            all_clas_numpy = all_clas.detach().cpu().numpy()[:, np.newaxis]

        ## aggregate the attacker's likelihood across k-fold cross validation
        total_lik = torch.zeros(1, device=self.device)
        total_acc = torch.zeros(1, device=self.device)
        for fold, (train_indices, test_indices) in enumerate(
            self.kf.split(all_scores_numpy, all_clas_numpy)
        ):
            train_indices = torch.from_numpy(train_indices)
            test_indices = torch.from_numpy(test_indices)
            train_indices.to(self.device), test_indices.to(self.device)
            X_tr, y_tr, y_clas_tr = (
                all_scores[train_indices],
                all_members[train_indices],
                all_clas[train_indices],
            )
            X_te, y_te, y_clas_te = (
                all_scores[test_indices],
                all_members[test_indices],
                all_clas[test_indices],
            )
            if not self.classwise:
                ## create a separate attacker to differentiate between forget&test instances for each class,
                ## as described in https://arxiv.org/abs/1610.05820
                unique_clas = torch.unique(y_clas_tr).detach().to(torch.long)
                for cl in unique_clas:
                    cls_idx_tr = torch.where(y_clas_tr == cl)
                    cls_idx_te = torch.where(y_clas_te == cl)
                    # attacker's likelihood and accuracy
                    att_lik, att_acc = self._attacker_likelihood(
                        X_tr[cls_idx_tr],
                        y_tr[cls_idx_tr],
                        X_te[cls_idx_te],
                        y_te[cls_idx_te],
                        classifier=self.att_classifier,
                    )
                    total_lik = total_lik + att_lik
                    total_acc = total_acc + att_acc
            else:
                att_lik, att_acc = self._attacker_likelihood(
                    X_tr, y_tr, X_te, y_te, classifier=self.att_classifier
                )
                total_lik = total_lik + att_lik
                total_acc = total_acc + att_acc
        return (
            total_lik / all_scores.size(0),
            total_acc / all_scores.size(0),
            wasserstein_dist,
        )

    def _attacker_likelihood(
        self, X_tr, y_tr, X_te, y_te, classifier="SVM"
    ) -> torch.Tensor:
        if classifier == "SVM":
            return self._attacker_likelihood_SVM(X_tr, y_tr, X_te, y_te)
        else:
            raise ValueError("Unsupported classifier for the attacker's problem.")

    def _attacker_likelihood_SVM(self, X_tr, y_tr, X_te, y_te) -> torch.Tensor:
        """
        Formulate the membership inference attack (MIA)
        as a differentiable layer of SVM
        """
        n_sample = X_tr.shape[0]
        n_feature = X_tr.shape[1]

        ## define the optimization problem of SVM in cvxpy
        beta = cp.Variable((n_feature, 1))
        b = cp.Variable()
        data = cp.Parameter((n_sample, n_feature))
        if int(torch.__version__[0]) > int(REF_VERSION[0]) or int(
            torch.__version__.split(".")[1]
        ) >= int(REF_VERSION.split(".")[1]):
            Y = 2 * y_tr.numpy(force=True)[:, np.newaxis] - 1
        else:
            Y = 2 * y_tr.detach().cpu().numpy()[:, np.newaxis] - 1
        ## margin loss
        loss = cp.sum(cp.pos(1 - cp.multiply(Y, data @ beta - b)))
        reg = self.attacker_reg * cp.norm(beta, 1)
        prob = cp.Problem(cp.Minimize(loss / n_sample + reg))
        attacker_layer = CvxpyLayer(prob, [data], [beta, b])
        ## run (X_tr, y_tr) through the attacker layer
        beta_tch, b_tch = attacker_layer(X_tr, solver_args={"solve_method": "SCS"})

        def hinge_loss(output, target):
            # For binary classification with labels +1 and -1
            return torch.clamp(1 - output * (2 * target - 1), min=0).sum()

        ## the attacker's utility, i.e., the negative of hinge loss
        t = X_te @ beta_tch - b_tch
        attacker_likelihood = -hinge_loss(t.squeeze(), y_te * 1.0)

        ## the attacker's accuracy
        with torch.no_grad():
            ## the forget data is labeld as 1
            preds = torch.where(
                t >= 0,
                torch.tensor(1, device=self.device),
                torch.tensor(0, device=self.device),
            )
            attacker_accuracy = (preds.squeeze() == y_te).sum().item()
        return (attacker_likelihood, attacker_accuracy)


if __name__ == "__main__":
    input_dim = 10
    num_samples = 500
    data = BinaryClassificationDataset(num_samples, input_dim)
    inputs, targets = data.data, data.labels
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    net = MLP(input_dim=10, hidden_dim=5, output_dim=2)
    net.to(DEVICE)
    num_class = 2
    bs = 100
    defender = DefenderOPT(
        DataLoader(data, batch_size=bs),
        DataLoader(data, batch_size=bs),
        DataLoader(data, batch_size=bs),
        DataLoader(data, batch_size=bs),
        num_class=2,
        defender_lr=0.01,
        attacker_lr=0.01,
        with_attacker=True,
        save_checkpoint=False,
        classwise=True,
    )
    t_start = time.time()
    defender.unlearn(net)
    t_end = time.time() - t_start
    print()
