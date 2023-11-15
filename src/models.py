import os
import sys
import time
import torch
import pickle
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# from utils import BinaryClassificationDataset
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

# import torchsort
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import pdb
from memory_profiler import profile

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
    def __init__(self, 
                 D_r: DataLoader = None, 
                 D_f: DataLoader = None, 
                 D_v: DataLoader = None,
                 D_t: DataLoader = None,
                 dim: int = 1,
                 cv: int = 5,
                 batch_size: int = 128,
                 num_epoch: int = 10,
                 baseline_mode: bool = False,
                 regular_type: str = 'l2',
                 regular_strength: float = 0.5,
                 num_class: int = 2,
                 attacker_lr: float = 0.01,
                 defender_lr: float = 0.01,
                 with_attacker: bool = True,
                 attacker_reg: float = 0.,
                 wasserstein_coeff: float = 0.5,
                 output_sc_fname: str = None,
                 seed: int = 42,
                 weight_decay: float = 5e-4,
                 momentum: float = 0.9,
                 device_id: int = 0,
                 fine_tune: bool = True,
                 att_classifier: str = 'SVM'):
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
        self.device = f'cuda:{device_id}' if torch.cuda.is_available() else "cpu"
        self.att_classifier = att_classifier

        ## whether to fine-tune on the retain set
        self.fine_tune = fine_tune

        ## keeps the attacker's optimization problems in memory
        ## for warm-start the solvers
        self.attacker_opt_cache = {}

        ## cross-validation split used in the attacker's optimization
        ## it's initialized here for having consistent number of train/test data
        self.kf = StratifiedShuffleSplit(n_splits=self.cv, test_size=0.3333, random_state=self.seed)    
        
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
        optimizer = optim.SGD(net.parameters(), 
                              lr=self.defender_lr, 
                              momentum=self.momentum,
                              weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epoch)
        
        ## check the performance before any optimization
        if self.baseline_mode:
            test_accuracy = self._evaluate_accuracy(net,   self.test_loader, device=self.device)
            forget_accuracy = self._evaluate_accuracy(net, self.forget_loader, device=self.device)
            retain_accuracy = self._evaluate_accuracy(net, self.retain_loader, device=self.device)
            MIA_accuracy, MIA_recall, MIA_auc = DefenderOPT._evaluate_MIA(net, 
                                                                          self.forget_loader, 
                                                                          self.val_loader, 
                                                                          dim=self.dim,
                                                                          device=self.device,
                                                                          seed=self.seed,
                                                                          output_sc_fname=self.output_sc_fname,
                                                                          save=True)
            print(f'test accuracy: {test_accuracy:.4f}, '
                  f'forget accuracy: {forget_accuracy:.4f}, '
                  f'retain accuracy: {retain_accuracy:.4f}, '
                  f'MIA accuracy: {MIA_accuracy.item():.4f}, '
                  f'MIA auc: {MIA_auc.item():.4f}, '
                  f'MIA recall: {MIA_recall.item():.4f}')
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
            test_accuracy = self._evaluate_accuracy(net,   self.test_loader, device=self.device)
            retain_accuracy = self._evaluate_accuracy(net, self.retain_loader, device=self.device)
            forget_accuracy = self._evaluate_accuracy(net, self.forget_loader, device=self.device)
            _save = True if epoch == self.num_epoch - 1 else False
            MIA_accuracy, MIA_recall, MIA_auc = DefenderOPT._evaluate_MIA(net, 
                                                                          self.forget_loader, 
                                                                          self.val_loader, 
                                                                          dim=self.dim,
                                                                          device=self.device,
                                                                          seed=self.seed,
                                                                          output_sc_fname=self.output_sc_fname,
                                                                          save=_save)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f'log_{self.seed}.txt'), 'a') as file_out:
                print(f'Epoch [{epoch+1}/{self.num_epoch}], ',
                      f'U_d: {u_d.item():.4f}, ',
                      f'U_a: {u_a.item():.4f}, ' if self.with_attacker else f'U_a: None, ',
                      f'test accuracy: {test_accuracy:.4f}, ',
                      f'forget accuracy: {forget_accuracy:.4f}, ',
                      f'retain accuracy: {retain_accuracy:.4f}, ',
                      f'att accuracy: {att_acc.item():.4f}, ' if self.with_attacker else f'att accuracy: None, ',
                      f'MIA accuracy: {MIA_accuracy.item():.4f}, ',
                      f'MIA auc: {MIA_auc.item():.4f}, ',
                      f'MIA recall: {MIA_recall.item():.4f}, ', file=file_out)
            print(f'time/epoch: {t_all:.4f} min; attacker opt: {t_att:.4f} ({t_att/t_all:.2f})')
            

    @staticmethod
    def _generate_scores(net, 
                         loader: DataLoader, 
                         mode='train', 
                         dim: int = 1,
                         device='cpu'):
        """
            Feed the test and forget sets through the unlearned model,
            and collect the output scores with the corresponding class labels.

            Parameters:
                loader: PyTorch data loader
            
            Return:
                scores: the score vectors with shape (n, dim)
                clas: class labels with shape (n, ). 
        """
        net.train() if mode == 'train' else net.eval()
        criterion = nn.CrossEntropyLoss(reduction="none")
        scores_list = []
        clas_list = []
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            losses = criterion(outputs, targets)
            new_score = losses[:, None] if dim == 1 \
                                        else torch.cat((outputs, losses[:, None]), axis=1)
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
            param_group['lr'] = new_lr    
    

    @staticmethod
    @torch.no_grad()
    def _evaluate_accuracy(net, 
                           eval_loader: DataLoader = None,
                           device='cpu'):
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
    def _evaluate_MIA(net,
                      forget_set: DataLoader = None,
                      test_set: DataLoader = None,
                      dim: int = 1,
                      device: str = 'cpu',
                      seed: int = 42,
                      output_sc_fname: str = "",
                      save: bool = False) -> np.array:
        """
            This function mimics the `simple_mia` function in the starting kit.
            Returns:
                MIA_accuracy (np.array): the averaged MIA accuracy.

        """
        ns = min(len(forget_set.dataset), len(test_set.dataset))
        forget_scores, _ = DefenderOPT._generate_scores(net, forget_set, mode='eval', dim=dim, device=device)
        test_scores, _   = DefenderOPT._generate_scores(net, test_set, mode='eval', dim=dim, device=device) # the naming is a bit bad here :(
        all_scores = torch.cat((forget_scores[:ns], test_scores[:ns]), axis=0) # shape=(ns, dim)
        forget_members, test_members = torch.ones(ns, device=device), torch.zeros(ns, device=device) 
        all_members = torch.cat((forget_members, test_members), axis=0) # shape=(ns, )

        if int(torch.__version__[0]) > int(REF_VERSION[0]) or int(torch.__version__.split('.')[1]) >= int(REF_VERSION.split('.')[1]):
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
        MIA_accuracy = cross_val_score(attack_model, all_scores, all_members, cv=cv, scoring="accuracy").mean()
        MIA_recall = cross_val_score(attack_model, all_scores, all_members, cv=cv, scoring="recall").mean()
        MIA_auc = cross_val_score(attack_model, all_scores, all_members, cv=cv, scoring="roc_auc").mean()
        
        ## save scores and members to disk
        if save:
            fname = f'output_scores' if not output_sc_fname \
                    else output_sc_fname
            with open(fname + '.p', 'wb') as fid:
                pickle.dump({'scores': all_scores[:, -1],  ## -1 since only include the scalar cross entropy scores
                             'members': all_members}, fid)
        return (MIA_accuracy, MIA_recall, MIA_auc)
        

    def _compute_1d_wasserstein_distance(self,
                                         ft_scores: torch.Tensor,
                                         tst_scores: torch.Tensor):
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


    def _attacker_opt(self, net = None):
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
        forget_scores, forget_clas = DefenderOPT._generate_scores(net, self.forget_loader, mode='train', dim=self.dim, device=self.device)
        test_scores, test_clas = DefenderOPT._generate_scores(net, self.val_loader, mode='train', dim=self.dim, device=self.device) # the naming is a bit bad here :(
        all_scores = torch.cat((forget_scores[:ns], test_scores[:ns]), axis=0) # shape=(ns, dim)
        all_clas = torch.cat((forget_clas[:ns], test_clas[:ns]), axis=0).to(torch.long) # shape=(ns, )
        forget_members, test_members = torch.ones(ns, device=self.device), torch.zeros(ns, device=self.device)

        ## compute 1d wasserstein distance
        # wasserstein_dist = self._compute_1d_wasserstein_distance(forget_scores, test_scores)
        wasserstein_dist = torch.tensor(0.0, device=self.device)

        all_members = torch.cat((forget_members, test_members), axis=0).to(torch.long)  # shape=(ns, )

        # shuffle the data
        idx = torch.randperm(all_scores.shape[0])
        all_scores = all_scores[idx]
        all_members = all_members[idx]
        all_clas = all_clas[idx]
        all_scores.to(self.device), all_members.to(self.device), all_clas.to(self.device)

        ## below is just to handle platform issues
        if int(torch.__version__[0]) > int(REF_VERSION[0]) or int(torch.__version__.split('.')[1]) >= int(REF_VERSION.split('.')[1]):
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
        for fold, (train_indices, test_indices) in enumerate(self.kf.split(all_scores_numpy, all_clas_numpy)):
            train_indices = torch.from_numpy(train_indices)
            test_indices = torch.from_numpy(test_indices)
            train_indices.to(self.device), test_indices.to(self.device)
            X_tr, y_tr, y_clas_tr = all_scores[train_indices], all_members[train_indices], all_clas[train_indices]
            X_te, y_te, y_clas_te = all_scores[test_indices], all_members[test_indices], all_clas[test_indices]
            ## create a separate attacker for each class, as described in https://arxiv.org/abs/1610.05820
            unique_clas = torch.unique(y_clas_tr).detach().to(torch.long)
            for cl in unique_clas:
                cls_idx_tr = torch.where(y_clas_tr == cl)
                cls_idx_te = torch.where(y_clas_te == cl)
                # attacker's likelihood and accuracy
                att_lik, att_acc = self._attacker_likelihood(X_tr[cls_idx_tr], 
                                                             y_tr[cls_idx_tr], 
                                                             X_te[cls_idx_te], 
                                                             y_te[cls_idx_te],
                                                             fold_id=fold,
                                                             class_id=cl.item(),
                                                             classifier=self.att_classifier)
                total_lik = total_lik + att_lik
                total_acc = total_acc + att_acc
        return (total_lik / all_scores.size(0), total_acc / all_scores.size(0), wasserstein_dist)
    

    def _attacker_likelihood(self, X_tr, y_tr, X_te, y_te, fold_id, class_id, classifier='SVM') -> torch.Tensor:
        if classifier == 'SVM':
            return self._attacker_likelihood_SVM(X_tr, y_tr, X_te, y_te, fold_id, class_id)
        elif classifier == 'LR':
            return self._attacker_likelihood_LR(X_tr, y_tr, X_te, y_te, fold_id, class_id)
        else:
            raise ValueError("Unsupported classifier for the attacker's problem.")
    
    
    def _attacker_likelihood_LR(self, X_tr, y_tr, X_te, y_te, fold_id, class_id) -> torch.Tensor:
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
            if int(torch.__version__[0]) > int(REF_VERSION[0]) or int(torch.__version__.split('.')[1]) >= int(REF_VERSION.split('.')[1]):
                Y = y_tr.numpy(force=True)[:, np.newaxis]
            else:
                Y = y_tr.detach().cpu().numpy()[:, np.newaxis]
            t = data @ beta + b
            loglik = (1. / n_sample) * cp.sum(
                cp.multiply(Y, t) - cp.logistic(t)
            )
            reg = -self.attacker_reg * cp.sum_squares(beta)
            prob = cp.Problem(cp.Maximize(loglik + reg))
            attacker_layer = CvxpyLayer(prob, [data], [beta, b])
            ## store `attacker_layer` in memory
            self.attacker_opt_cache[(fold_id, class_id)] = [attacker_layer]
            ## run (X_tr, y_tr) through the attacker layer
            beta_tch, b_tch = attacker_layer(X_tr, solver_args={'solve_method':'SCS'})
        else:
            attacker_layer = self.attacker_opt_cache[(fold_id, class_id)][0]
            beta_tch, b_tch = attacker_layer(X_tr, solver_args={'solve_method':'SCS'})


        ## will average later
        loss_func = nn.BCEWithLogitsLoss(reduction='sum') 
        ## the attacker's utility, i.e., the negative of the cross-entropy loss 
        t = X_te @ beta_tch + b_tch
        attacker_likelihood = -loss_func(t.squeeze(), y_te*1.0)
        ## the attacker's accuracy
        with torch.no_grad():
            probs = torch.sigmoid(t)
            ## the forget data is labeld as 1
            preds = torch.where(probs >= 0.5, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device))
            attacker_accuracy = (preds.squeeze() == y_te).sum().item()
        return (attacker_likelihood, attacker_accuracy)

        
    def _attacker_likelihood_SVM(self, X_tr, y_tr, X_te, y_te, fold_id, class_id) -> torch.Tensor:
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
            if int(torch.__version__[0]) > int(REF_VERSION[0]) or int(torch.__version__.split('.')[1]) >= int(REF_VERSION.split('.')[1]):
                Y = y_tr.numpy(force=True)[:, np.newaxis]
            else:
                Y = y_tr.detach().cpu().numpy()[:, np.newaxis]
            ## margin loss 
            loss = cp.sum(cp.pos(1 - cp.multiply(Y, data @ beta - b)))
            reg = self.attacker_reg * cp.norm(beta, 1)
            prob = cp.Problem(cp.Minimize(loss/n_sample + reg))
            attacker_layer = CvxpyLayer(prob, [data], [beta, b])
            ## store `attacker_layer` in memory
            self.attacker_opt_cache[(fold_id, class_id)] = [attacker_layer]
            ## run (X_tr, y_tr) through the attacker layer
            beta_tch, b_tch = attacker_layer(X_tr, solver_args={'solve_method':'SCS'})
        else:
            attacker_layer = self.attacker_opt_cache[(fold_id, class_id)][0]
            beta_tch, b_tch = attacker_layer(X_tr, solver_args={'solve_method':'SCS'})

        def hinge_loss(output, target):
            # For binary classification with labels +1 and -1
            return torch.clamp(1 - output * (2*target - 1), min=0).sum()
        
        ## the attacker's utility, i.e., the negative of hinge loss
        t = X_te @ beta_tch - b_tch
        attacker_likelihood = -hinge_loss(t.squeeze(), y_te*1.0)

        ## the attacker's accuracy
        with torch.no_grad():
            ## the forget data is labeld as 1
            preds = torch.where(t >= 0, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device))
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




class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, imagenet=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        print('The normalize layer is contained in the network')
        self.normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

        if not imagenet:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        # See note [TorchScript super()]
        x = self.normalize(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

