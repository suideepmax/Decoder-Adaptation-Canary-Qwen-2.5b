"""AdamW with fp32 master weights and fp32 optimizer state, for use under
mixed-precision (fp16) training without a framework-managed GradScaler and
without fp32 master-weight support in the base optimizer.

Motivation: under plain `torch.optim.AdamW` applied directly to fp16
parameters, `exp_avg_sq` underflows to exactly zero at this model's
realistic gradient magnitudes (on the order of 1e-5 squared), collapsing
the AdamW update into sign-based SGD-with-momentum at an effective
learning rate of `lr/eps`, not the intended adaptive update. This also
makes `weight_decay` numerically inert in fp16, since the decoupled-decay
factor `1 - lr*weight_decay` rounds to exactly 1.0 whenever
`lr*weight_decay` is below fp16's representable precision near 1.0
(roughly 4.9e-4) -- true for the learning rates and weight-decay values
commonly used for this model.

This optimizer keeps fp32 shadow copies of each parameter, its first
moment (`exp_avg`), and its second moment (`exp_avg_sq`), and performs the
full AdamW update in fp32 regardless of the model's fp16 storage dtype.
All operations are elementwise, so this is compatible with FSDP2's
DTensor (sharded) parameters without any rank-changing reshapes.

Loss-scale handling: incoming gradients in `p.grad` are assumed to have
been computed from a loss already multiplied by `loss_scale` upstream.
`loss_scale` is a mutable attribute, so a training loop can implement
dynamic loss scaling (halve on overflow, grow after N consecutive clean
steps) without reconstructing the optimizer.

Gradient clipping: this optimizer divides the incoming gradient by
`grad_divisor`, not `loss_scale` directly. `grad_divisor` defaults to
`loss_scale` (plain unscaling, no clipping), but a caller can fold in a
global-norm clip coefficient by setting `grad_divisor = loss_scale /
clip_coef` before calling `step()`. This makes gradient clipping part of
this optimizer's existing fp32 divide, rather than a separate operation
applied to the raw fp16 gradient tensor -- clipping a fp16 tensor
directly (e.g. via a framework-native clip call operating on
unconverted, sharded gradients) is a distinct correctness hazard under
FSDP2, and multiplying an already-fp16 gradient by a small fp32 clip
coefficient and writing the result back to fp16 risks collapsing most
elements to exactly zero on that write-back.
"""

import itertools
import warnings
from collections import defaultdict

import torch


class MasterWeightAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0, loss_scale=1024.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.loss_scale = float(loss_scale)
        self.grad_divisor = float(loss_scale)
        self._had_overflow_this_step = False

    # --- Persistent numerical state (survives resume) --------------------
    # loss_scale/grad_divisor are the dynamic loss-scale state, mutated
    # every step by the training loop's backoff/growth logic -- genuinely
    # persistent state, deliberately distinct from transient per-step
    # flags like _had_overflow_this_step, which are correctly reset on
    # every resume and are not included here.
    #
    # The default PyTorch Optimizer.load_state_dict() silently downcasts
    # every floating-point per-parameter state tensor to the owning
    # parameter's storage dtype. Since this optimizer's parameters are
    # fp16, that means state["master"], state["exp_avg"], and
    # state["exp_avg_sq"] -- saved as fp32 by design, the entire purpose
    # of this class -- would be silently cast back to fp16 on every
    # resume, regardless of this class's own save/restore logic. At this
    # model's realistic per-element magnitudes, that downcast makes
    # exp_avg_sq underflow to exactly fp16 zero, and the AdamW epsilon
    # term underflows to zero as well, producing a division by exactly
    # zero (and therefore -inf) on the very first optimizer step after
    # resume. A post-hoc cast back to fp32 cannot undo this: the
    # precision, and the exact zero values, are already lost by the time
    # any code could intervene. The fix is to prevent the downcast from
    # happening at all, by not routing these three state keys through the
    # base class's per-parameter casting logic.
    _FP32_STATE_KEYS = ("master", "exp_avg", "exp_avg_sq")

    def state_dict(self):
        sd = super().state_dict()
        sd["loss_scale"] = self.loss_scale
        sd["grad_divisor"] = self.grad_divisor
        return sd

    def load_state_dict(self, state_dict):
        # Deliberately does not call super().load_state_dict() -- that
        # method is exactly what silently downcasts master/exp_avg/
        # exp_avg_sq to fp16 (see class docstring above), and the cast is
        # unconditional rather than gated by the incoming tensor's own
        # dtype, so it cannot be prevented via a pre/post hook. This
        # reimplements the same structural validation and param-id
        # remapping the base class performs, while preserving fp32 for
        # this optimizer's own state tensors.
        state_dict = dict(state_dict)  # shallow copy; do not mutate caller's dict
        if "loss_scale" not in state_dict:
            warnings.warn(
                "MasterWeightAdamW.load_state_dict: checkpoint has no saved 'loss_scale' -- "
                f"falling back to this run's configured default ({self.loss_scale}). This is "
                "NOT an exact resume of the dynamic loss-scale state.", stacklevel=2,
            )
        loss_scale = state_dict.pop("loss_scale", self.loss_scale)
        grad_divisor = state_dict.pop("grad_divisor", self.grad_divisor)

        groups = self.param_groups
        saved_groups = state_dict["param_groups"]
        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        for g, sg in zip(groups, saved_groups):
            if len(g["params"]) != len(sg["params"]):
                raise ValueError(
                    "loaded state dict contains a parameter group that doesn't match "
                    "the size of optimizer's group"
                )

        id_map = dict(zip(
            itertools.chain.from_iterable(g["params"] for g in saved_groups),
            itertools.chain.from_iterable(g["params"] for g in groups),
        ))

        new_state = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k not in id_map:
                new_state[k] = v  # unassociated state, kept as-is (base class does the same)
                continue
            param = id_map[k]
            entry = {}
            for key, val in v.items():
                if key in self._FP32_STATE_KEYS and torch.is_tensor(val):
                    # Preserve fp32 exactly. Deliberately does not also
                    # force a device move here: by the time this method
                    # runs, the incoming tensor has already been placed
                    # onto this rank's own local (sharded) device by the
                    # upstream checkpoint-loading machinery, so forcing a
                    # device change can move another rank's already-correct
                    # local state onto the wrong device under FSDP2. Only
                    # the dtype is changed, and only if not already fp32.
                    entry[key] = val if val.dtype == torch.float32 else val.to(dtype=torch.float32)
                else:
                    entry[key] = val
            new_state[param] = entry

        new_param_groups = []
        for g, sg in zip(groups, saved_groups):
            ng = dict(sg)
            ng["params"] = g["params"]
            new_param_groups.append(ng)

        self.__setstate__({"state": new_state, "param_groups": new_param_groups})
        self.loss_scale = float(loss_scale)
        self.grad_divisor = float(grad_divisor)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._had_overflow_this_step = False

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # No per-parameter finiteness check here: on a sharded
                # DTensor gradient that check is itself a collective
                # operation, and skipping it only on affected ranks risks
                # a distributed-communication hang. The caller is expected
                # to perform one global, correctly-all-reduced finiteness
                # check upstream and zero all gradients identically on
                # every rank before this ever runs, so gradients reaching
                # this optimizer are always already finite.
                grad = p.grad.detach()
                grad = grad.float() / self.grad_divisor

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["master"] = p.detach().clone().float()
                    state["exp_avg"] = torch.zeros_like(state["master"])
                    state["exp_avg_sq"] = torch.zeros_like(state["master"])

                state["step"] += 1
                step = state["step"]
                master = state["master"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if wd != 0.0:
                    master.mul_(1 - lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                step_size = lr / bias_correction1

                master.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.copy_(master.to(p.dtype))

        return loss
