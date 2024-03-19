import logging
import os
import pdb
import pickle
import sys
import time

# import torchsort
import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cvxpylayers.torch import CvxpyLayer
from memory_profiler import profile
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

# from utils import BinaryClassificationDataset
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

# self.device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ATTACKER = 0
REF_VERSION = "1.13.1"


# An MLP class for debugging purposes
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
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
        output_sc_fname: str = None,
        seed: int = 42,
        weight_decay: float = 5e-4,
        momentum: float = 0.9,
        device_id: int = 0,
        fine_tune: bool = True,
        att_classifier: str = "SVM",
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
        self.output_sc_fname = output_sc_fname
        self.seed = seed
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.att_classifier = att_classifier

        ## whether to fine-tune on the retain set
        self.fine_tune = fine_tune

        ## keeps the attacker's optimization problems in memory
        ## for warm-start the solvers
        self.attacker_opt_cache = {}

        ## cross-validation split used in the attacker's optimization
        ## it's initialized here for having consistent number of train/test data
        self.kf = StratifiedShuffleSplit(
            n_splits=self.cv, test_size=0.3333, random_state=self.seed
        )

    def unlearn(self, net, output_dir):
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
                output_sc_fname=self.output_sc_fname,
                save=True,
            )
            print(
                f"test accuracy: {test_accuracy:.4f}, "
                f"forget accuracy: {forget_accuracy:.4f}, "
                f"retain accuracy: {retain_accuracy:.4f}, "
                f"MIA accuracy: {MIA_accuracy.item():.4f}, "
                f"MIA auc: {MIA_auc.item():.4f}, "
                f"MIA recall: {MIA_recall.item():.4f}"
            )
            return net

        net.train()
        for epoch in range(self.num_epoch):
            t_start = time.time()
            ## fine-tune on the retain set
            if self.fine_tune:
                for inputs, targets in self.retain_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    u_d = -loss_func(outputs, targets)
                    (-u_d).backward()
                    optimizer.step()

            t_att_start = time.time()
            if self.with_attacker:
                ## the attacker's utility
                ## _set_lr enables different lr for the defender and attacker
                self._set_lr(optimizer, new_lr=self.attacker_lr)
                att_lik, att_acc, _ = self._attacker_opt(net)
                u_a = att_lik
                optimizer.zero_grad()
                ## the defender wants to minimize the attacker's utility u_a
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
            _save = True if epoch == self.num_epoch - 1 else False
            MIA_accuracy, MIA_recall, MIA_auc = DefenderOPT._evaluate_MIA(
                net,
                self.forget_loader,
                self.val_loader,
                dim=self.dim,
                device=self.device,
                seed=self.seed,
                output_sc_fname=self.output_sc_fname,
                save=_save,
            )
            os.makedirs(output_dir, exist_ok=True)
            with open(
                os.path.join(output_dir, f"log_{self.seed}.txt"), "a"
            ) as file_out:
                print(
                    f"Epoch [{epoch+1}/{self.num_epoch}], ",
                    f"U_d: {u_d.item():.4f}, ",
                    (
                        f"U_a: {u_a.item():.4f}, "
                        if self.with_attacker
                        else f"U_a: None, "
                    ),
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
                    file=file_out,
                )
            print(
                f"time/epoch: {t_all:.4f} min; attacker opt: {t_att:.4f} ({t_att/t_all:.2f})"
            )

    @staticmethod
    def _generate_scores(
        net, loader: DataLoader, mode="train", dim: int = 1, device="cpu"
    ):
        """
        Feed the test and forget sets through the unlearned model,
        and collect the output scores with the corresponding class labels.

        Parameters:
            loader: PyTorch data loader

        Return:
            scores: the score vectors with shape (n, dim)
            clas: class labels with shape (n, ).
        """
        net.train() if mode == "train" else net.eval()
        criterion = nn.CrossEntropyLoss(reduction="none")
        scores_list = []
        clas_list = []
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            losses = criterion(outputs, targets)
            new_score = (
                losses[:, None]
                if dim == 1
                else torch.cat((outputs, losses[:, None]), axis=1)
            )
            scores_list.append(new_score)
            clas_list.append(targets)
        scores = torch.cat(scores_list, axis=0)
        clas = torch.cat(clas_list, axis=0)
        return (scores, clas)

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
        for inputs, targets in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
        acc = total_correct / total_samples
        return acc

    @staticmethod
    @torch.no_grad()
    def _evaluate_MIA(
        net,
        forget_set: DataLoader = None,
        test_set: DataLoader = None,
        dim: int = 1,
        device: str = "cpu",
        seed: int = 42,
        output_sc_fname: str = "",
        save: bool = False,
    ) -> np.array:
        """
        This function mimics the `simple_mia` function in the starting kit.
        Returns:
            MIA_accuracy (np.array): the averaged MIA accuracy.

        """
        ns = min(len(forget_set.dataset), len(test_set.dataset))
        forget_scores, _ = DefenderOPT._generate_scores(
            net, forget_set, mode="eval", dim=dim, device=device
        )
        test_scores, _ = DefenderOPT._generate_scores(
            net, test_set, mode="eval", dim=dim, device=device
        )  # the naming is a bit bad here :(
        all_scores = torch.cat(
            (forget_scores[:ns], test_scores[:ns]), axis=0
        )  # shape=(ns, dim)
        forget_members, test_members = torch.ones(ns, device=device), torch.zeros(
            ns, device=device
        )
        all_members = torch.cat((forget_members, test_members), axis=0)  # shape=(ns, )

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

        attack_model = LogisticRegression()
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

        ## save scores and members to disk
        if save:
            fname = f"output_scores" if not output_sc_fname else output_sc_fname
            with open(fname + ".p", "wb") as fid:
                pickle.dump(
                    {
                        "scores": all_scores[
                            :, -1
                        ],  ## -1 since only include the scalar cross entropy scores
                        "members": all_members,
                    },
                    fid,
                )
        return (MIA_accuracy, MIA_recall, MIA_auc)

    def _compute_1d_wasserstein_distance(
        self, ft_scores: torch.Tensor, tst_scores: torch.Tensor
    ):
        """
        Compute the 1d Wasserstein distance between the output scalar scores.
        Parameters:
            ft_scores/tst_scores: the scores of the data from the forget/test set
        Returns:
            dist: 1d wasserstein distance
        """
        # _diff_sort_func = lambda x: torchsort.soft_sort(x,
        #                                                 regularization=self.regular_type,
        #                                                 regularization_strength=self.regular_strength)
        # ft_scores.to(self.device), tst_scores.to(self.device)
        # return torch.mean(torch.abs(
        #    _diff_sort_func(ft_scores) - _diff_sort_func(tst_scores)
        #     ))
        return torch.tensor(0.0, device=self.device)

    def _attacker_opt(self, net=None):
        """
        The attacker's utility maximizing problem.

        Parameters:
            net: the unlearned model

        Returns:
            total_ua: the attacker's utility summed across the folds.

        """
        assert net is not None, "The input net is None.\n"
        ## the scores and memberships for test and forget sets
        ## NOTICE: len(dataloader) != len(dataset)
        ns = min(len(self.forget_loader.dataset), len(self.val_loader.dataset))
        forget_scores, forget_clas = DefenderOPT._generate_scores(
            net, self.forget_loader, mode="train", dim=self.dim, device=self.device
        )
        test_scores, test_clas = DefenderOPT._generate_scores(
            net, self.val_loader, mode="train", dim=self.dim, device=self.device
        )  # the naming is a bit bad here :(
        all_scores = torch.cat(
            (forget_scores[:ns], test_scores[:ns]), axis=0
        )  # shape=(ns, dim)
        all_clas = torch.cat((forget_clas[:ns], test_clas[:ns]), axis=0).to(
            torch.long
        )  # shape=(ns, )
        forget_members, test_members = torch.ones(ns, device=self.device), torch.zeros(
            ns, device=self.device
        )

        ## compute 1d wasserstein distance
        # wasserstein_dist = self._compute_1d_wasserstein_distance(forget_scores, test_scores)
        wasserstein_dist = torch.tensor(0.0, device=self.device)

        all_members = torch.cat((forget_members, test_members), axis=0).to(
            torch.long
        )  # shape=(ns, )

        # shuffle the data
        idx = torch.randperm(all_scores.shape[0])
        all_scores = all_scores[idx]
        all_members = all_members[idx]
        all_clas = all_clas[idx]
        all_scores.to(self.device), all_members.to(self.device), all_clas.to(
            self.device
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
        ## self.kf = StratifiedShuffleSplit.
        ## It is important to have a fixed ratio of each class for both training and test data,
        ## as we need to warm-start the solver.
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
            ## create a separate attacker for each class, as described in https://arxiv.org/abs/1610.05820
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
                    fold_id=fold,
                    class_id=cl.item(),
                    classifier=self.att_classifier,
                )
                total_lik = total_lik + att_lik
                total_acc = total_acc + att_acc
        return (
            total_lik / all_scores.size(0),
            total_acc / all_scores.size(0),
            wasserstein_dist,
        )

    def _attacker_likelihood(
        self, X_tr, y_tr, X_te, y_te, fold_id, class_id, classifier="SVM"
    ) -> torch.Tensor:
        if classifier == "SVM":
            return self._attacker_likelihood_SVM(
                X_tr, y_tr, X_te, y_te, fold_id, class_id
            )
        elif classifier == "LR":
            return self._attacker_likelihood_LR(
                X_tr, y_tr, X_te, y_te, fold_id, class_id
            )
        else:
            raise ValueError("Unsupported classifier for the attacker's problem.")

    def _attacker_likelihood_LR(
        self, X_tr, y_tr, X_te, y_te, fold_id, class_id
    ) -> torch.Tensor:
        """
        Formulate the membership inference attack (MIA)
        as a differentiable layer of Logistic Regression (LR)

        The input data includes:
            1) X_tr/X_te: score outputs X from the unlearned model, which can be (n, 1) (when the scores are scalars)
                          or (n, k) (when the scores are vectors)
            2) y_tr/y_te: labels indicating membership of test/forget sets; shape=(n, )
        Parameters:
            (X_tr, y_tr): data to train the LR
            (X_te, y_te): data to test how good the LR is
            (fold_id, class_id): identifies the attacker's optimization problem
                                 for the particular combination of (fold, clas)

        Returns:
            Ua: the attacker's likelihood
        """
        ## the first time to initiate an attacker
        if (fold_id, class_id) not in self.attacker_opt_cache:
            n_sample = X_tr.shape[0]
            n_feature = X_tr.shape[1]

            ## define the optimization of logistic regression in cvxpy
            beta = cp.Variable((n_feature, 1))
            b = cp.Variable((1, 1))
            data = cp.Parameter((n_sample, n_feature))
            if int(torch.__version__[0]) > int(REF_VERSION[0]) or int(
                torch.__version__.split(".")[1]
            ) >= int(REF_VERSION.split(".")[1]):
                Y = y_tr.numpy(force=True)[:, np.newaxis]
            else:
                Y = y_tr.detach().cpu().numpy()[:, np.newaxis]
            t = data @ beta + b
            loglik = (1.0 / n_sample) * cp.sum(cp.multiply(Y, t) - cp.logistic(t))
            reg = -self.attacker_reg * cp.sum_squares(beta)
            prob = cp.Problem(cp.Maximize(loglik + reg))
            attacker_layer = CvxpyLayer(prob, [data], [beta, b])
            ## store `attacker_layer` in memory
            self.attacker_opt_cache[(fold_id, class_id)] = [attacker_layer]
            ## run (X_tr, y_tr) through the attacker layer
            beta_tch, b_tch = attacker_layer(X_tr, solver_args={"solve_method": "SCS"})
        else:
            attacker_layer = self.attacker_opt_cache[(fold_id, class_id)][0]
            beta_tch, b_tch = attacker_layer(X_tr, solver_args={"solve_method": "SCS"})

        ## will average later
        loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        ## the attacker's utility, i.e., the negative of the cross-entropy loss
        t = X_te @ beta_tch + b_tch
        attacker_likelihood = -loss_func(t.squeeze(), y_te * 1.0)
        ## the attacker's accuracy
        with torch.no_grad():
            probs = torch.sigmoid(t)
            ## the forget data is labeld as 1
            preds = torch.where(
                probs >= 0.5,
                torch.tensor(1, device=self.device),
                torch.tensor(0, device=self.device),
            )
            attacker_accuracy = (preds.squeeze() == y_te).sum().item()
        return (attacker_likelihood, attacker_accuracy)

    def _attacker_likelihood_SVM(
        self, X_tr, y_tr, X_te, y_te, fold_id, class_id
    ) -> torch.Tensor:
        """
        Formulate the membership inference attack (MIA)
        as a differentiable layer of support vector machine (SVM)
        """
        if (fold_id, class_id) not in self.attacker_opt_cache:
            n_sample = X_tr.shape[0]
            n_feature = X_tr.shape[1]

            ## define the optimization of logistic regression in cvxpy
            beta = cp.Variable((n_feature, 1))
            b = cp.Variable()
            data = cp.Parameter((n_sample, n_feature))
            if int(torch.__version__[0]) > int(REF_VERSION[0]) or int(
                torch.__version__.split(".")[1]
            ) >= int(REF_VERSION.split(".")[1]):
                Y = y_tr.numpy(force=True)[:, np.newaxis]
            else:
                Y = y_tr.detach().cpu().numpy()[:, np.newaxis]
            ## margin loss
            loss = cp.sum(cp.pos(1 - cp.multiply(Y, data @ beta - b)))
            reg = self.attacker_reg * cp.norm(beta, 1)
            prob = cp.Problem(cp.Minimize(loss / n_sample + reg))
            attacker_layer = CvxpyLayer(prob, [data], [beta, b])
            ## store `attacker_layer` in memory
            self.attacker_opt_cache[(fold_id, class_id)] = [attacker_layer]
            ## run (X_tr, y_tr) through the attacker layer
            beta_tch, b_tch = attacker_layer(X_tr, solver_args={"solve_method": "SCS"})
        else:
            attacker_layer = self.attacker_opt_cache[(fold_id, class_id)][0]
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


# if __name__ == "__main__":
#     input_dim = 10
#     num_samples = 500
#     data = BinaryClassificationDataset(num_samples, input_dim)
#     inputs, targets = data.data, data.labels
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#     net = MLP(input_dim=10,
#               hidden_dim=5,
#               output_dim=2)
#     net.to(DEVICE)
#     num_class = 2
#     bs = 1
#     defender = DefenderOPT(DataLoader(data, batch_size=bs),
#                            DataLoader(data, batch_size=bs),
#                            DataLoader(data, batch_size=bs),
#                            DataLoader(data, batch_size=bs),
#                            num_class=2,
#                            defender_lr=0.01,
#                            attacker_lr=0.01,
#                            with_attacker=True)
#     defender.unlearn(net)
#     print()