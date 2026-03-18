#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, override

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.models import GCN, GAT, GraphSAGE, MLP

from src import dataset

type PropMethod = Literal['SM', 'SK', 'SP']


@dataclass(slots=True)
class AttackLogEntry:
    name:      str
    seed:      int | None
    flip_frac: float
    n_train:   int
    c_max:     int
    n_flipped: int
    extras:    dict[str, Any]


class Attack:
    def __init__(self, seed: int | None, flip_frac: float):
        self.seed      = seed
        self.flip_frac = flip_frac
        self.log: list[AttackLogEntry] = []

    def _get_generator(self, device) -> torch.Generator:
        gen = torch.Generator(device=device)
        if self.seed is not None:
            gen.manual_seed(self.seed)
        return gen

    def _validate_inputs(self, data: Data, mask: torch.BoolTensor) -> None:
        if not mask.any():
            raise ValueError("mask contains no training nodes")
        if data.y is None:
            raise ValueError("data.y must not be None")

    def _compute_budget(self, mask: torch.BoolTensor) -> tuple[int, int]:
        n_train = int(mask.sum().item())
        c_max   = max(1, round(self.flip_frac * n_train))
        return n_train, c_max

    def _write_log(
        self,
        flip_frac: float,
        n_train:   int,
        c_max:     int,
        n_flipped: int,
        **kwargs:  Any,
    ) -> None:
        self.log.append(AttackLogEntry(
            name=self.__class__.__name__,
            seed=self.seed,
            flip_frac=flip_frac,
            n_train=n_train,
            c_max=c_max,
            n_flipped=n_flipped,
            extras=kwargs,
        ))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(seed={self.seed!r}, flip_frac={self.flip_frac!r})"

    def init_attack(self, data: Data, mask: torch.BoolTensor) -> Data:
        return data

    def update_attack(self, data: Data, mask: torch.BoolTensor, **kwargs) -> Data:
        return data


class NoAttack(Attack):
    def __init__(self, seed: int | None, flip_frac: float = 0.0):
        super().__init__(seed, flip_frac)

    @override
    def init_attack(self, data: Data, mask: torch.BoolTensor) -> Data:
        return dataset.clone_data(data)

    @override
    def update_attack(self, data: Data, mask: torch.BoolTensor, **kwargs) -> Data:
        return data


class RandomFlipAttack(Attack):
    def __init__(self, seed: int | None, flip_frac: float):
        super().__init__(seed, flip_frac)

    @override
    def init_attack(self, data: Data, mask: torch.BoolTensor) -> Data:
        self._validate_inputs(data, mask)
        idx = mask.nonzero(as_tuple=False).view(-1)
        n_train, num_flip = self._compute_budget(mask)

        gen      = self._get_generator(idx.device)
        idx_perm = torch.randperm(len(idx), generator=gen, device=idx.device)
        flip_idx = idx[idx_perm[:num_flip]]

        y          = data.y.clone()
        num_labels = int(y.max().item()) + 1

        new_labels = y[flip_idx].clone()
        same       = torch.ones(num_flip, dtype=torch.bool, device=y.device)
        # ensure that we have different labels for every node
        while same.any():
            new_labels[same] = torch.randint(
                0, num_labels, (same.sum().item(),), generator=gen, device=y.device
            )
            same = new_labels == y[flip_idx]
        y[flip_idx] = new_labels

        poisoned_data   = dataset.clone_data(data)
        poisoned_data.y = y

        self._write_log(self.flip_frac, n_train, num_flip, num_flip)

        return poisoned_data

    @override
    def update_attack(self, data: Data, mask: torch.BoolTensor, **kwargs) -> Data:
        return data  # return as is


class DegreeFlipAttack(Attack):
    def __init__(self, seed: int | None, flip_frac: float):
        super().__init__(seed, flip_frac)

    @override
    def init_attack(self, data: Data, mask: torch.BoolTensor) -> Data:
        self._validate_inputs(data, mask)
        edge_index = data.edge_index
        n_train, num_flip = self._compute_budget(mask)
        # calculate degree = out edges + in edges
        net_degree = degree(edge_index[1]) + degree(edge_index[0])
        deg, flip_idx = torch.topk(net_degree[mask], k=num_flip, largest=True)

        y          = data.y.clone()
        num_labels = int(y.max().item()) + 1

        gen        = self._get_generator(y.device)
        new_labels = y[flip_idx].clone()
        same       = torch.ones(num_flip, dtype=torch.bool, device=y.device)
        # ensure that we have different labels for every node
        while same.any():
            new_labels[same] = torch.randint(
                0, num_labels, (same.sum().item(),), generator=gen, device=y.device
            )
            same = new_labels == y[flip_idx]
        y[flip_idx] = new_labels

        poisoned_data   = dataset.clone_data(data)
        poisoned_data.y = y

        self._write_log(self.flip_frac, n_train, num_flip, num_flip)

        return poisoned_data

    @override
    def update_attack(self, data: Data, mask: torch.BoolTensor, **kwargs) -> Data:
        return data  # return as is


class LafAKAttack(Attack):
    """
    LafAK: Adversarial Label-Flipping Attack for GNNs.
    Zhang et al., "Adversarial Label-Flipping Attack and Defense for
    Graph Neural Networks", ICDM 2020.

    Poisoning attack that flips a small fraction of training labels so that
    a GCN retrained on the poisoned data suffers maximum misclassification.

    The bi-level optimisation (outer: find best flips; inner: retrain GCN) is
    collapsed to a single level by replacing the GCN with a closed-form linear
    approximation (paper Section IV-B, Eq. 7-8).  All work therefore happens
    once, before training, in `init_attack`.  `update_attack` is a no-op.
    """

    def __init__(
        self,
        seed:         int | None,
        flip_frac:    float,
        target_label: int | tuple | None = None,
        atk_epochs:   int   = 200,
        gcn_l2:       float = 5e-4,
        lr:           float = 1e-4,
    ):
        """
        Parameters
        ----------
        seed         : RNG seed for reproducibility.
        flip_frac    : ε — fraction of training labels allowed to flip
                       (paper Eq. 5: ‖δ − 1‖₀ ≤ ε · N_L).
        target_label : optional (a, b) tuple to override automatic class
                       selection; rarely needed.
        atk_epochs   : SGD iterations for optimising α (Algorithm 1, lines 3-7).
        gcn_l2       : λ — L2 regularisation in the OLS closed form
                       (paper Eq. 7: ‖Â²Xθ_L − y_L‖² + λ‖θ‖²).
        lr           : SGD learning rate for the α update.
        """
        super().__init__(seed, flip_frac)
        self.target_label = target_label
        self.atk_epochs   = atk_epochs
        self.gcn_l2       = gcn_l2
        self.lr           = lr

        self.tau: int | None = None   # set by _select_tau; exposed for inspection

        # Candidate smoothness coefficients for the tanh surrogate of sign(·).
        # Paper Section IV-C: "The larger the value of τ, the closer the smooth
        # function is to the step function." (also see Fig. 3)
        self.tau_list = [1, 2, 4, 8, 16, 32, 64, 128]


    # ----------------------------------------------------------------------- #
    #  Public interface                                                        #
    # ----------------------------------------------------------------------- #

    @override
    def init_attack(self, data: Data, mask: torch.BoolTensor) -> Data:
        """
        Run the full LafAK attack and return a poisoned copy of `data`.

        Follows Algorithm 1 (paper Section IV-D):
          1. Resolve which two classes to attack.
          2. Pre-compute K (closed-form GCN surrogate, Eq. 8).
          3. Search for the best τ and optimise α via SGD (lines 3-7).
          4. Flip the top-c_max nodes by α (line 8).
          5. Map binary flips back to the original multi-class labels.

        Parameters
        ----------
        data         : PyG Data object — left unmodified.
        mask         : boolean mask identifying training nodes.
        """
        self._validate_inputs(data, mask)

        a, b = self._resolve_classes(data, mask, self.target_label)

        train_mask, unlabeled_mask = self._binary_masks(data, mask, a, b)
        train_idx     = train_mask.nonzero(as_tuple=True)[0]
        unlabeled_idx = unlabeled_mask.nonzero(as_tuple=True)[0]

        # Labels in {-1, +1} required by the attack objective.
        # Paper Section IV-A: y ∈ {-1, +1}^N
        y_signed    = self._to_signed(data.y, a)
        y_train     = y_signed[train_idx]
        y_unlabeled = y_signed[unlabeled_idx]

        # Attack budget: c_max = ε · N_L  (paper Eq. 5)
        n_train, c_max = self._compute_budget(mask)

        # Closed-form GCN surrogate (paper Section IV-B, Eq. 8)
        K = self._compute_K(data, train_idx, unlabeled_idx)

        # Optimise α and select flipped nodes
        flip_local = self._select_tau(K, y_train, y_unlabeled, c_max)

        # Apply flips — only to training labels, test/val are never written to.
        new_data        = data.clone()
        flip_global_idx = train_idx[flip_local]
        for idx in flip_global_idx:
            lbl = int(new_data.y[idx].item())
            # Swap class a ↔ class b to recover original label space.
            # Paper binaryAttack_multiclass step 4:
            # "change the perturbed binary labels back to multi-class labels"
            new_data.y[idx] = b if lbl == a else a

        n_flipped = int(flip_local.sum().item())
        self._write_log(self.flip_frac, n_train, c_max, n_flipped, a=a, b=b, tau=self.tau)

        return new_data

    @override
    def update_attack(self, data: Data, mask: torch.BoolTensor, **kwargs) -> Data:
        """
        No-op: LafAK is a pure poisoning attack.

        Because the bi-level problem is collapsed to single-level via the
        closed-form GCN approximation (paper Eq. 8-9), all label flips are
        determined once in `init_attack` and the attack needs no access to
        model state during training.
        """
        return data


    # ----------------------------------------------------------------------- #
    #  Step 1 — Class selection                                                #
    # ----------------------------------------------------------------------- #

    def _resolve_classes(self, data, mask, target_label):
        """
        Identify the two classes (a, b) to use in the binary attack.

        Paper Section VI-A: "Our attacks focus on semi-supervised binary
        classification.  For multi-class datasets, we use LafAK to generate
        attacks for the two classes with the most nodes."
        """
        n_classes = int(data.y.max().item()) + 1

        if n_classes == 2:
            return 0, 1

        if target_label is not None:
            if isinstance(target_label, int):
                return target_label, (target_label + 1) % n_classes
            return int(target_label[0]), int(target_label[1])

        counts = torch.bincount(data.y[mask], minlength=n_classes)
        top2   = counts.topk(2).indices
        return int(top2[0]), int(top2[1])

    def _binary_masks(self, data, mask, a, b):
        """Return (train_ab_mask, unlabeled_ab_mask) restricted to classes a and b."""
        ab_mask         = (data.y == a) | (data.y == b)
        train_mask      = mask & ab_mask
        unlabeled_mask  = ~mask & ab_mask
        return train_mask, unlabeled_mask

    @staticmethod
    def _to_signed(y, neg_class):
        """Map class labels to {-1, +1}: neg_class → -1, everything else → +1."""
        return torch.where(
            y == neg_class,
            torch.tensor(-1.0, device=y.device),
            torch.tensor(+1.0, device=y.device),
        )


    # ----------------------------------------------------------------------- #
    #  Step 2 — Closed-form GCN surrogate (K matrix)                          #
    # ----------------------------------------------------------------------- #

    def _compute_K(self, data, train_idx, unlabeled_idx):
        """
        Compute K such that  ŷ_U = K · y_L  gives the predicted unlabeled
        labels under the linearised, OLS-solved GCN.

        Paper Section IV-B, Eq. (8):
          P  = ((Â²X)_L^T (Â²X)_L + λI)^{-1} (Â²X)_L^T
          K  = (Â²X)_U · P       shape: [N_U, N_L]
        """
        A2X = self._propagate_features(data)   # Â²X, shape [N, d]

        A2X_L = A2X[train_idx]                 # [N_L, d]
        A2X_U = A2X[unlabeled_idx]             # [N_U, d]

        d    = A2X_L.shape[1]
        gram = A2X_L.T @ A2X_L + self.gcn_l2 * torch.eye(d, device=A2X_L.device)
        P    = torch.linalg.pinv(gram) @ A2X_L.T   # [d, N_L]
        K    = A2X_U @ P                            # [N_U, N_L]
        return K

    def _propagate_features(self, data):
        """
        Two-hop feature propagation: X̄ = Â²X.

        Linearisation step of paper Section IV-B, Eq. (6):
          Softmax(Â(ÂXW^(1))W^(2)) ≈ Softmax(Â²X·W)
        (non-linearities between layers are dropped, as in SGC [ref 28]).

        Â is the symmetrically normalised adjacency with self-loops
        (paper Section III-A, Eq. 1): Â = D̃^{-1/2} Ã D̃^{-1/2}, Ã = A + I.
        """
        n      = data.num_nodes
        A_hat  = self._normalised_adjacency(data.edge_index, n)

        # Row-normalise features (matches row_normalize in original code)
        X      = data.x.float()
        X_norm = X / X.sum(dim=1, keepdim=True).clamp(min=1e-8)

        AX  = torch.sparse.mm(A_hat, X_norm)   # Â X
        A2X = torch.sparse.mm(A_hat, AX)       # Â² X
        return A2X

    @staticmethod
    def _normalised_adjacency(edge_index, n):
        """Sparse Â = D̃^{-1/2} Ã D̃^{-1/2} with self-loops."""
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=n)
        deg              = degree(edge_index_sl[0], num_nodes=n)
        d_inv_sqrt       = deg.pow(-0.5).clamp(max=1e9)
        row, col         = edge_index_sl
        vals             = d_inv_sqrt[row] * d_inv_sqrt[col]
        return torch.sparse_coo_tensor(
            edge_index_sl, vals, (n, n), device=edge_index.device
        ).coalesce()


    # ----------------------------------------------------------------------- #
    #  Step 3 — τ search and α optimisation                                   #
    # ----------------------------------------------------------------------- #

    def _select_tau(self, K, y_train, y_unlabeled, c_max):
        """
        Iterate over TAU_LIST, run the gradient attack for each τ, and keep
        the τ that achieves the lowest closed-form attack accuracy.

        Paper Section IV-C (get_tau in original code):
        "select the hyperparameter τ (larger τ, the closer to the step function)."
        """
        best_tau  = self.tau_list[0]
        best_acc  = float("inf")
        best_flip = None

        for tau in self.tau_list:
            alpha     = self._optimise_alpha(K, y_train, y_unlabeled, c_max, tau)
            flip_mask = self._greedy_flip_mask(alpha, c_max)

            y_l_atk            = y_train.float().clone()
            y_l_atk[flip_mask] *= -1
            acc = self._closed_form_accuracy(K, y_l_atk, y_unlabeled)

            if acc <= best_acc:
                best_acc  = acc
                best_tau  = tau
                best_flip = flip_mask

        self.tau = best_tau
        return best_flip

    def _optimise_alpha(self, K, y_train, y_unlabeled, c_max, tau):
        """
        Optimise the Bernoulli flip-probability vector α via SGD.

        Implements Algorithm 1 lines 3-7 (paper Section IV-D).

        Objective (paper Eq. 12):
          L(α) = -E[1/N_U Σ_i ỹ_U[i] · y_U[i]]

        where (paper Eq. 10):
          ỹ_U = tanh(τ · K · (y_L ⊙ z))

        z is the continuous relaxation of δ ∈ {-1,+1} sampled via the
        Gumbel reparameterisation trick (paper Section IV-C, citing
        Figurnov et al. NeurIPS 2018 [ref 39]).  The temperature 0.5 and
        the mapping z = 2/(1+tmp)-1 match the original TF implementation.
        """
        n_L = y_train.shape[0]
        dev = K.device
        y_l = y_train.float().unsqueeze(1)       # [N_L, 1]
        y_u = y_unlabeled.float().unsqueeze(1)   # [N_U, 1]

        # α initialised to 0.5 (original: initial_value=0.5*np.ones_like(y_l))
        alpha = torch.full((n_L, 1), 0.5, device=dev, requires_grad=True)
        opt   = torch.optim.SGD([alpha], lr=self.lr)

        gen = self._get_generator(K.device)

        for _ in range(self.atk_epochs):
            opt.zero_grad()

            eps = self._gumbel_sample(n_L, dev, gen) \
                - self._gumbel_sample(n_L, dev, gen)

            a_clamped = alpha.clamp(1e-6, 1 - 1e-6)
            logit_a   = torch.log(a_clamped / (1 - a_clamped))

            tmp = torch.exp((logit_a + eps) / 0.5)
            z   = 2.0 / (1.0 + tmp) - 1.0          # z ∈ [-1, 1]

            y_l_perturbed = y_l * z                                     # y_L ⊙ z
            y_u_pred      = torch.tanh(tau * (K @ y_l_perturbed))       # ỹ_U

            # Minimise mean(ỹ_U · y_U) ⟺ maximise misclassification (Eq. 12)
            loss = torch.mean(y_u_pred * y_u)
            loss.backward()
            opt.step()

        return alpha.detach().squeeze(1)   # [N_L]

    @staticmethod
    def _gumbel_sample(n, device, generator):
        """Sample n i.i.d. Gumbel(0,1) variates."""
        u = torch.zeros(n, 1, device=device).uniform_(1e-8, 1 - 1e-8,
                                                       generator=generator)
        return -(-u.log()).log()


    # ----------------------------------------------------------------------- #
    #  Step 4 — Greedy flip selection                                          #
    # ----------------------------------------------------------------------- #

    def _greedy_flip_mask(self, alpha, c_max):
        """
        Select up to c_max nodes to flip, choosing those with the highest α.

        Paper Section IV-D, line 8: "flip the labels in y'_L with the largest
        N_L elements in α."  Nodes are only flipped if α[i] > 0.5, which is
        described as "similar to ε-greedy strategy [ref 14]."
        """
        idx       = torch.argsort(alpha, descending=True)
        flip_mask = torch.zeros(len(alpha), dtype=torch.bool, device=alpha.device)
        count     = 0
        for i in idx:
            if alpha[i] > 0.5 and count < c_max:
                flip_mask[i] = True
                count += 1
            if count >= c_max:
                break
        return flip_mask


    # ----------------------------------------------------------------------- #
    #  Evaluation utility                                                      #
    # ----------------------------------------------------------------------- #

    @staticmethod
    def _closed_form_accuracy(K, y_train, y_unlabeled):
        """
        Evaluate closed-form GCN accuracy on unlabeled nodes.

        Paper Section IV-B, Eq. (8): ŷ_U = K · y_L.
        Accuracy = mean(sign(ŷ_U) == y_U).
        Matches closedForm_bin in the original code.
        """
        y_pred = K @ y_train.float().unsqueeze(1)
        return (torch.sign(y_pred.squeeze()) == y_unlabeled.float()).float().mean().item()


class MGAttack(Attack):
    """
    MG Attack: Adversarial Label Poisoning Attack on GNNs via Label Propagation.

    Liu et al., "Adversarial Label Poisoning Attack on Graph Neural Networks
    via Label Propagation", ECML-PKDD 2022.
    
    Gradient-based label poisoning attack using label propagation as a GCN
    surrogate.

    The key insight (paper Section 3) is that, due to the equivalence between
    decoupled GCN and label propagation [ref 4], attacking only the LP model
    is as effective as attacking the full GCN, while requiring no knowledge of
    neural network parameters.

    Three propagation strategies (paper Section 3.2) are available for both
    the attack objective (Ā) and pseudo-label generation:
      SM  Gaussian kernel, graph-structure-agnostic  (Sec 3.2a, Eq. 7)
      SK  Normalised adjacency Â^K                   (Sec 3.2b, SGCN [ref 30])
      SP  APPNP propagation α(I-(1-α)Â)^{-1}         (Sec 3.2c, APPNP [ref 11])
    """

    PROP_METHODS = ('SM', 'SK', 'SP')

    def __init__(
        self,
        seed:           int | None,
        flip_frac:      float,
        n_iter:         int        = 2,
        attack_prop:    PropMethod = 'SM',
        pred_prop:      PropMethod = 'SK',
        gamma:          float      = 1.0,
        pagerank_alpha: float      = 0.1,
        prop_K:         int        = 2,
    ):
        """
        Parameters
        ----------
        seed            : RNG seed (currently unused; present for interface parity).
        flip_frac       : fraction of training labels to poison
                          (budget c = round(flip_frac * n_l)).
        n_iter          : N — MG attack iterations (Algorithm 1).
                          Paper Figure 1 caption: "N=2 is a good choice in practice."
        attack_prop     : Ā propagation method for the MG gradient (paper Sec 3.2).
        pred_prop       : LP method used to generate pseudo-labels ŷu.
                          Paper Sec 4: SK (ŷu = Â^K yl) is the default.
        gamma           : γ in Gaussian kernel S_ij = exp(-γ||xi-xj||²) for SM.
        pagerank_alpha  : α teleport probability for SP (paper Eq. 3, APPNP [ref 11]).
        prop_K          : K hops for SK (Â^K) and SP power iterations.
        """
        assert attack_prop in self.PROP_METHODS
        assert pred_prop   in self.PROP_METHODS

        super().__init__(seed, flip_frac)
        self.n_iter         = n_iter
        self.attack_prop    = attack_prop
        self.pred_prop      = pred_prop
        self.gamma          = gamma
        self.pagerank_alpha = pagerank_alpha
        self.prop_K         = prop_K


    # ----------------------------------------------------------------------- #
    #  Public interface                                                        #
    # ----------------------------------------------------------------------- #

    @override
    def init_attack(self, data: Data, mask: torch.BoolTensor) -> Data:
        """
        Run the full MG attack and return a poisoned copy of `data`.

        Implements the three-component framework (paper Figure 1, Section 3):
          1. Label propagation  — generate pseudo-labels ŷu (Section 3.1).
          2. Maximum gradient attack — identify training labels to poison (Alg 1).
          3. GCN training with poisoned labels — return poisoned data (Section 3.3).

        Unlike LafAK, this attack is natively multi-class and does not require
        any GCN model parameters (black-box attack, paper Section 5 comparison).

        Parameters
        ----------
        data         : PyG Data object — left unmodified.
        mask         : boolean mask identifying training nodes.
        """
        self._validate_inputs(data, mask)

        train_idx     = mask.nonzero(as_tuple=True)[0]
        unlabeled_idx = (~mask).nonzero(as_tuple=True)[0]

        n_train, c_max = self._compute_budget(mask)
        n_classes = int(data.y.max().item()) + 1

        # One-hot labels for labeled nodes — needed for matrix-form LP (Sec 3.2).
        Y_l = F.one_hot(data.y[train_idx], n_classes).float()   # [n_l, C]

        # ---- Step 1: pseudo-label generation (paper Section 3.1) ---- #
        A_ul_pred = self._compute_A_ul(data, train_idx, unlabeled_idx, self.pred_prop)
        Y_u_hat   = A_ul_pred @ Y_l    # ŷu = Ā y_l,  shape [n_u, C]

        # ---- Step 2: MG attack (Algorithm 1, paper Section 3.2) ---- #
        A_ul_atk = self._compute_A_ul(data, train_idx, unlabeled_idx, self.attack_prop)

        # Majority class for the flip target.
        # Paper Algorithm 1 line 7: "Modify y^t_l[i] for all i ∈ I to the max
        # label class"; Section 5 (LafAK_c comparison) confirms this means the
        # globally most-frequent class in the dataset.
        majority_class = int(torch.bincount(data.y).argmax().item())

        flip_local = self._run_mg_attack(A_ul_atk, Y_l, Y_u_hat, c_max,
                                         majority_class, n_classes)

        # ---- Step 3: apply flips — training labels only ---- #
        # Test/validation labels are never written to; they are only read as
        # ŷu to guide the attack objective (same design decision as LafAK).
        new_data = data.clone()
        new_data.y[train_idx[flip_local]] = majority_class

        self._write_log(
            self.flip_frac, n_train, c_max, len(flip_local),
            attack_prop=self.attack_prop,
            pred_prop=self.pred_prop,
            majority_class=majority_class,
        )

        return new_data

    @override
    def update_attack(self, data: Data, mask: torch.BoolTensor, **kwargs) -> Data:
        """
        No-op: MG is a pure poisoning attack, all decisions made in init_attack.

        Paper Section 3.3: poisoned labels are fixed before GCN training begins;
        no intermediate model state is needed.
        """
        return data


    # ----------------------------------------------------------------------- #
    #  Step 1 — Propagation matrix A_ul                                       #
    # ----------------------------------------------------------------------- #

    def _compute_A_ul(
        self,
        data: Data,
        train_idx: torch.Tensor,
        unlabeled_idx: torch.Tensor,
        method: str,
    ) -> torch.Tensor:
        """
        Build the [n_u, n_l] propagation sub-matrix A_ul for the chosen method.

        A_ul encodes how labeled node labels propagate to unlabeled nodes:
          ŷu = A_ul Y_l

        Paper Section 3.2 defines three strategies:
          SM  A_ul = (D_uu - S_uu)^{-1} S_ul   (Eq. 7; Gaussian kernel)
          SK  A_ul = (Â^K)[unlabeled, labeled]   (SGCN [ref 30])
          SP  A_ul = (α(I-(1-α)Â)^{-1})[unlabeled, labeled]  (APPNP [ref 11])
        """
        if method == 'SM':
            return self._sm_A_ul(data.x, train_idx, unlabeled_idx)
        elif method == 'SK':
            return self._sk_A_ul(data.edge_index, data.num_nodes, train_idx, unlabeled_idx)
        else:  # SP
            return self._sp_A_ul(data.edge_index, data.num_nodes, train_idx, unlabeled_idx)

    def _sm_A_ul(
        self,
        X: torch.Tensor,
        train_idx: torch.Tensor,
        unlabeled_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        SM: A_ul = (D_uu - S_uu)^{-1} S_ul  (paper Eq. 7, Section 3.2a).

        S_ij = exp(-γ ||x_i - x_j||²) — Gaussian kernel similarity.
        D: degree matrix with D_ii = Σ_j S_ij over all nodes.
        (D_uu - S_uu) is the graph Laplacian block for unlabeled nodes.

        This is the only method that does not use graph structure — it relies
        solely on node features. Paper Section 5 (hypothesis 2): "graph
        structural information is not necessary" in the presence of features.

        NOTE: requires O(n_u × n_l) + O(n_u²) memory. For graphs with more
        than ~5,000 nodes the pairwise distance computation may be expensive.
        """
        X_u = X[unlabeled_idx].float()   # [n_u, d]
        X_l = X[train_idx].float()       # [n_l, d]

        # S_ul: [n_u, n_l] — similarity between unlabeled and labeled nodes
        diff_ul = X_u.unsqueeze(1) - X_l.unsqueeze(0)    # [n_u, n_l, d]
        S_ul    = torch.exp(-self.gamma * (diff_ul ** 2).sum(-1))

        # S_uu: [n_u, n_u] — similarity among unlabeled nodes
        diff_uu = X_u.unsqueeze(1) - X_u.unsqueeze(0)    # [n_u, n_u, d]
        S_uu    = torch.exp(-self.gamma * (diff_uu ** 2).sum(-1))

        # D_uu diagonal: D_ii = Σ_{j∈unlabeled} S_ij + Σ_{j∈labeled} S_ij
        D_uu_diag = S_uu.sum(dim=1) + S_ul.sum(dim=1)    # [n_u]
        L_uu      = torch.diag(D_uu_diag) - S_uu         # D_uu - S_uu, [n_u, n_u]

        # Solve (D_uu - S_uu) A_ul = S_ul  →  A_ul = L_uu^{-1} S_ul
        A_ul = torch.linalg.solve(L_uu + 1e-8 * torch.eye(len(unlabeled_idx),
                                                            device=X.device), S_ul)
        return A_ul   # [n_u, n_l]

    def _sk_A_ul(
        self,
        edge_index: torch.Tensor,
        n: int,
        train_idx: torch.Tensor,
        unlabeled_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        SK: A_ul = (Â^K)[unlabeled_idx, train_idx]  (paper Section 3.2b, [ref 30]).

        Computed by propagating labeled-node indicator vectors K hops through Â.
        """
        A_hat = self._normalised_adjacency(edge_index, n)

        n_l = len(train_idx)
        dev = edge_index.device

        # E_l: [n, n_l] — column j is the indicator vector for train_idx[j]
        E_l = torch.zeros(n, n_l, device=dev)
        E_l[train_idx, torch.arange(n_l, device=dev)] = 1.0

        # Apply Â^K iteratively (paper Eq. 4: Ā = Â^K for SGCN)
        H = E_l
        for _ in range(self.prop_K):
            H = torch.sparse.mm(A_hat, H)   # [n, n_l]

        return H[unlabeled_idx]   # [n_u, n_l]

    def _sp_A_ul(
        self,
        edge_index: torch.Tensor,
        n: int,
        train_idx: torch.Tensor,
        unlabeled_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        SP: A_ul = (α(I-(1-α)Â)^{-1})[unlabeled_idx, train_idx]  (paper Sec 3.2c).

        Approximated via APPNP power iteration [ref 11]:
          H^(0)   = E_l
          H^(k+1) = (1-α) Â H^(k) + α E_l
        which converges to α(I-(1-α)Â)^{-1} E_l as K → ∞.

        Paper Eq. (3): Ā = (1-α)^K Â^K + α Σ_{k=0}^{K-1} (1-α)^k Â^k
        """
        A_hat = self._normalised_adjacency(edge_index, n)
        n_l   = len(train_idx)
        dev   = edge_index.device

        E_l = torch.zeros(n, n_l, device=dev)
        E_l[train_idx, torch.arange(n_l, device=dev)] = 1.0

        H = E_l.clone()
        for _ in range(self.prop_K):
            H = (1 - self.pagerank_alpha) * torch.sparse.mm(A_hat, H) \
                + self.pagerank_alpha * E_l

        return H[unlabeled_idx]   # [n_u, n_l]


    # ----------------------------------------------------------------------- #
    #  Step 2 — MG attack iterations (Algorithm 1)                            #
    # ----------------------------------------------------------------------- #

    def _run_mg_attack(
        self,
        A_ul: torch.Tensor,
        Y_l: torch.Tensor,
        Y_u_hat: torch.Tensor,
        c_max: int,
        majority_class: int,
        n_classes: int,
    ) -> torch.Tensor:
        """
        Algorithm 1 from paper (Section 3.2): N iterations of gradient-based
        label poisoning.

        Objective (Eq. 8):
          f(Y^t_l) = -1/2 ||Ā Y^t_l - Ŷu||²_F
        Gradient:
          ∂f/∂Y^t_l = Ā^T (Ŷu - Ā Y^t_l),   shape [n_l, C]
        Per-node score:
          s[i] = ||∂f/∂Y^t_l[i,:]||₂          (line 5)

        Each iteration:
          1. Compute gradient on Y^t_l (current poisoned labels).
          2. Select top-c nodes by gradient magnitude             (line 5).
          3. Reset Y^t_l = Y_l (original), flip selected to majority class
             (lines 6-7), producing Y^{t+1}_l.

        Returns local indices (within train_idx) of nodes flipped in the
        final iteration.
        """
        Y_orig = Y_l.clone()   # y_l — original labels, never modified
        Y_curr = Y_l.clone()   # y^t_l — updated each iteration

        majority_onehot = F.one_hot(
            torch.tensor(majority_class, device=A_ul.device), n_classes
        ).float()

        flip_local = None

        for _ in range(self.n_iter):
            # Line 4: ∂f/∂Y^t_l = Ā^T (Ŷu - Ā Y^t_l)
            residual = Y_u_hat - A_ul @ Y_curr           # [n_u, C]
            G        = A_ul.T @ residual                 # [n_l, C]

            # Line 5: select top-c nodes by ||g[i,:]||₂
            scores     = G.norm(dim=1)                   # [n_l]
            flip_local = scores.topk(min(c_max, len(scores))).indices

            # Lines 6-7: reset to original, then apply flips
            Y_curr                = Y_orig.clone()
            Y_curr[flip_local]    = majority_onehot      # flip to max label class

        return flip_local


    # ----------------------------------------------------------------------- #
    #  Graph utility                                                           #
    # ----------------------------------------------------------------------- #

    @staticmethod
    def _normalised_adjacency(edge_index: torch.Tensor, n: int) -> torch.Tensor:
        """
        Â = D̃^{-1/2} Ã D̃^{-1/2},  Ã = A + I.

        Used by SK and SP propagation (paper Section 2.1, Eq. 1, [ref 10]).
        Identical preprocessing to LafAK's _normalised_adjacency.
        """
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=n)
        deg        = degree(edge_index_sl[0], num_nodes=n)
        d_inv_sqrt = deg.pow(-0.5).clamp(max=1e9)
        row, col   = edge_index_sl
        vals       = d_inv_sqrt[row] * d_inv_sqrt[col]
        return torch.sparse_coo_tensor(
            edge_index_sl, vals, (n, n), device=edge_index.device
        ).coalesce()


def build_attack(
    name: str,
    seed: int | None,
    flip_frac: float,
    target_label: int | tuple | None = None,
    lafak_atk_epochs: int = 200,
    lafak_gcn_l2: float = 5e-4,
    lafak_lr: float = 1e-4,
    mg_n_iter: int = 2,
    mg_attack_prop: PropMethod = "SK",
    mg_pred_prop: PropMethod = "SK",
    mg_gamma: float = 1.0,
    mg_pagerank_alpha: float = 0.1,
    mg_prop_k: int = 2,
) -> Attack:
    key = name.strip().lower()
    if key in {"none", "no_attack", "baseline", "clean"}:
        return NoAttack(seed=seed, flip_frac=0.0)
    if key in {"label_flipping", "random", "random_flip", "random_flipping"}:
        return RandomFlipAttack(seed=seed, flip_frac=flip_frac)
    if key in {"degree_flipping", "degree", "degree_flip"}:
        return DegreeFlipAttack(seed=seed, flip_frac=flip_frac)
    if key in {"lafak", "lafak_attack"}:
        return LafAKAttack(
            seed=seed,
            flip_frac=flip_frac,
            target_label=target_label,
            atk_epochs=lafak_atk_epochs,
            gcn_l2=lafak_gcn_l2,
            lr=lafak_lr,
        )
    if key in {"mg", "mg_attack"}:
        return MGAttack(
            seed=seed,
            flip_frac=flip_frac,
            n_iter=mg_n_iter,
            attack_prop=mg_attack_prop,
            pred_prop=mg_pred_prop,
            gamma=mg_gamma,
            pagerank_alpha=mg_pagerank_alpha,
            prop_K=mg_prop_k,
        )

    raise ValueError(f"unknown attack name: {name!r}")
