# copy dependencies from transformers/optimization.py
import gc
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from transformers.utils.versions import require_version

from loro_torch.lowrank_module import LowRankLinear
from loro_torch.lowrank_adpt_module import LowRankAdapterLinear


def get_loro_update_fn(type):
    if type == "eucl":

        def loro_update_fn(optimizer, M, N, dM, dN, use_exact_loro):

            lr_M = optimizer.state[M]["lr"]
            wd_M = optimizer.state[M]["wd"]

            lr_N = optimizer.state[N]["lr"]
            wd_N = optimizer.state[N]["wd"]

            M -= lr_M * dM
            N -= lr_N * dN

            if wd_M > 0:
                M -= lr_M * wd_M * M
            if wd_N > 0:
                N -= lr_N * wd_N * N

            return M, N

        return loro_update_fn

    elif type == "loro":

        def loro_update_fn(optimizer, M, N, dM, dN, use_exact_loro):

            lr_M = optimizer.state[M]["lr"]
            lr_mul_M = optimizer.state[M]["lr_scaler"]
            wd_M = optimizer.state[M]["wd"]

            lr_N = optimizer.state[N]["lr"]
            lr_mul_N = optimizer.state[N]["lr_scaler"]
            wd_N = optimizer.state[N]["wd"]

            assert lr_M == lr_N
            lr = lr_M
            lr_mul = (lr_mul_M + lr_mul_N) * 0.5

            # NOTE: lazy LORO update
            if not use_exact_loro:

                M -= lr_M * lr_mul_M * dM
                N -= lr_N * lr_mul_N * dN

                if wd_M > 0:
                    M -= lr_M * wd_M * M
                if wd_N > 0:
                    N -= lr_N * wd_N * N

                return M, N

            # NOTE: exact LORO update

            n, r = M.shape
            dtype = M.dtype
            M, N = M.float(), N.float()
            dM, dN = dM.float(), dN.float()

            """
            1. Project grad dW to tangent space at W
            W = U @ S @ V.T
            G_proj = U @ U.T @ G @ (I - V @ V.T)
                    + (I - U @ U.T) @ G @ V @ V.T
                    + U @ U.T @ G @ V @ V.T

            W - lr * G_proj = U @ (U.T @ G @ V + S) @ V.T
                        + [(I - U @ U.T) @ G @ V] @ V.T
                        + U @ [U.T @ G @ (I - V @ V.T)]

            Y1 = [U.T @ G @ (I - V @ V.T)].T
            Y2 = [(I - U @ U.T) @ G @ V]

            Q1, K1 = QR(Y1)
            Q1, K2 = QR(Y2)
            K0 = U.T @ G @ V
            """
            U, Rm = torch.linalg.qr(M)
            V, Rn = torch.linalg.qr(N)

            G_x_V = torch.linalg.solve(Rn.T, dM.T).T  # dM @ Rn_inv
            Ut_x_G = torch.linalg.solve(Rm.T, dN.T)  # Rm_inv.T @ dN.T

            K0 = Ut_x_G @ V
            Y1 = (Ut_x_G - Ut_x_G @ V @ V.T).T
            Y2 = G_x_V - U @ U.T @ G_x_V
            del G_x_V, Ut_x_G

            Q1, K1 = torch.linalg.qr(Y1)
            Q2, K2 = torch.linalg.qr(Y2)

            """
            2. Retract W - lr * G_proj to the manifold
            """
            Sigma_row1 = torch.cat([K0, K1.T], dim=1)
            Sigma_row2 = torch.cat([K2, torch.zeros(r, r).to(K0)], dim=1)
            del K0, K1, K2
            Sigma = torch.cat([Sigma_row1, Sigma_row2], dim=0)
            del Sigma_row1, Sigma_row2

            S_aug = torch.zeros(2 * r, 2 * r).to(Sigma)
            S_aug[:r, :r] = Rm @ Rn.T

            U_sig, S_sig, V_sig = torch.svd(S_aug - lr * Sigma)
            del Sigma, S_aug
            S_sig_sqrt = S_sig.pow(0.5).diag()[:, :r]
            M_new = torch.cat([U, Q2], dim=1) @ U_sig @ S_sig_sqrt
            N_new = torch.cat([V, Q1], dim=1) @ V_sig @ S_sig_sqrt
            del U_sig, S_sig, V_sig, U, V, Q1, Q2

            return M_new.to(dtype), N_new.to(dtype)

        return loro_update_fn
    else:
        raise ValueError(f"Invalid LORO update type: {type}")


class LOROAdamW(Optimizer):

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        loro_type: str = None,
        model: nn.Module = None,
    ):
        assert model is not None
        self.model = model

        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

        self.loro_type = loro_type
        self.loro_update_fn = get_loro_update_fn(loro_type)

    def refresh_states(self, refresh_mode):
        to_refresh = {
            "reg": ["regular"],
            "lrk": ["lowrank_in", "lowrank_out"],
            "all": ["lowrank_in", "lowrank_out", "regular"],
        }

        for group in self.param_groups:
            if group["type"] in to_refresh[refresh_mode]:
                for p in group["params"]:
                    state = self.state[p]
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

    def _regular_step(self, group):

        for i, p in enumerate(group["params"]):
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError(
                    "Adam does not support sparse gradients, please consider SparseAdam instead"
                )

            state = self.state[p]

            if "step" not in state:
                state["step"] = 0

            # State initialization
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(grad)
                state["exp_avg_sq"] = torch.zeros_like(grad)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            beta1, beta2 = group["betas"]

            state["step"] += 1

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            denom = exp_avg_sq.sqrt().add_(group["eps"])

            step_size = group["lr"]

            if group["correct_bias"]:  # No bias correction for Bert
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

            # compute norm gradient
            norm_grad = exp_avg / denom
            p.add_(norm_grad, alpha=-step_size)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want to decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            # Add weight decay at the end (fixed version)
            if group["weight_decay"] > 0.0:
                p.add_(
                    p,
                    alpha=(-group["lr"] * group["weight_decay"]),
                )

    def _calc_lowrank_grad(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError(
                    "Adam does not support sparse gradients, please consider SparseAdam instead"
                )

            state = self.state[p]

            # state initialization

            if "step" not in state:
                state["step"] = 0

            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(grad)
                state["exp_avg_sq"] = torch.zeros_like(grad)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            beta1, beta2 = group["betas"]

            state["step"] += 1

            exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            denom = exp_avg_sq.sqrt().add_(group["eps"])

            # update lowrank grad
            p.grad = exp_avg / denom

            step_size = group["lr"]
            if group["correct_bias"]:  # No bias correction for Bert
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

            lr_scaler = float(group.get("lr_scaler", 1.0))
            if lr_scaler < 0:
                rank = min(p.shape)
                hidden_size = group.get("hidden_size", max(p.shape))
                lr_scaler = rank / hidden_size

            # store stats for LORO update
            self.state[p]["lr"] = step_size
            self.state[p]["lr_scaler"] = lr_scaler
            self.state[p]["wd"] = group["weight_decay"]

    def _loro_step(self, use_exact_loro):
        for module in self.model.modules():
            if isinstance(module, (LowRankLinear, LowRankAdapterLinear)):
                M, N = self.loro_update_fn(
                    optimizer=self,
                    M=module.weight_out,
                    N=module.weight_in,
                    dM=module.weight_out.grad,
                    dN=module.weight_in.grad,
                    use_exact_loro=use_exact_loro,
                )
                module.weight_out.data.copy_(M)
                module.weight_in.data.copy_(N)
                del M, N

    @torch.no_grad()
    def step(self, closure: Callable = None, use_exact_loro=None):
        assert use_exact_loro is not None

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group["type"] == "regular":
                self._regular_step(group)
            elif group["type"] in ["lowrank_in", "lowrank_out"]:
                self._calc_lowrank_grad(group)
            else:
                raise ValueError(f"Invalid parameter group type: {group['type']}")

        self._loro_step(use_exact_loro=use_exact_loro)

        return loss
