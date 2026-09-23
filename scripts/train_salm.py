"""Training entrypoint for Canary-Qwen fine-tuning under fp16 (`precision:
16-true`) with numerically stable optimization. This is a thin wrapper
around NeMo's standard SALM training loop: the model is a StableSALM
subclass adding four things on top of NeMo's own `SALM`:

1. Dynamic loss scaling. The training loss is multiplied by a running
   `loss_scale` before backward, so fp16 gradients do not underflow to
   zero before the optimizer sees them. The scale backs off by 0.5x on
   any non-finite gradient and grows by 2.0x after a configurable number
   of consecutive clean steps, mirroring the standard
   `torch.amp.GradScaler` backoff/growth policy.

2. A correctness fix for gradient-norm computation under FSDP2/DTensor
   sharded parameters. Framework-native gradient clipping computes each
   rank's local gradient-shard norm in fp16 before the cross-rank
   all-reduce; at this model's realistic gradient magnitudes, squaring a
   fp16 value before reducing can overflow to `inf`, silently collapsing
   the clip coefficient to exactly zero and zeroing every gradient on
   that step. This is fixed by computing the global gradient norm as a
   single fp32 all-reduce over the *local* shards (never touching a
   sharded DTensor with a collective op that could partially desync
   ranks), and folding both the loss-scale division and the clip
   coefficient into one fp32 divide performed inside the optimizer
   itself (`MasterWeightAdamW`, see `master_weight_adamw.py`).

3. A one-time correctness tripwire: after the first non-skipped optimizer
   step, verify that at least one trainable parameter's optimizer state
   (`exp_avg`) is actually non-zero, i.e. that a real update occurred.
   This catches the specific failure mode above (gradients silently
   zeroed between measurement and use) at the start of a run rather than
   discovering it only after a checkpoint has already trained for hours
   without changing.

4. A sustained-skip tripwire: if more than 50 consecutive optimizer steps
   are skipped (non-finite gradients), the run raises rather than
   continuing to consume compute on a model that has likely diverged.

`MasterWeightAdamW` (same directory) keeps an fp32 master copy of every
trainable parameter and performs the AdamW update in fp32, dividing the
incoming fp16 gradient by a single scalar `grad_divisor` (loss scale,
or loss scale over clip coefficient on a clean step) that this file sets
every step. That attribute is the single source of truth for the
optimizer's division; this file never keeps a separate copy.

Also implemented here: loading the released model's own pretrained
speech-text connector ("bridge") and encoder weights before fine-tuning
begins (`model.load_released_pretrained: true`). By default, composing a
SALM from its published base components (`pretrained_llm` +
`pretrained_asr`) gives a randomly-initialized bridge layer and the raw,
unadapted encoder checkpoint, not the released model's own further-tuned
versions of either. This is an easily-missed configuration default in
NeMo's SALM composition path, not specific to this codebase -- the model
publisher has documented that fine-tuning from the released checkpoint
requires setting `pretrained_weights: False` and manually loading its
weights, since the default config path does not do this automatically.
This module implements that manual load, including correct handling for
LoRA-wrapped target models (mapping the released model's plain weight
keys onto the LoRA-wrapped model's parameter names, leaving the local
LoRA adapters at their standard PEFT cold-start initialization) and for
FSDP2's sharded (DTensor) parameters, which require each loaded tensor to
be re-sharded to match the model's existing device mesh and placement
before assignment.

Usage: `torchrun --nproc_per_node=<N> train_salm.py --config-path=<dir>
--config-name=<config>`, with a config using `master_weight_adamw.
MasterWeightAdamW` as the optimizer target.
"""

import json
import os
import re
import time

import hydra
import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from torch.distributed.tensor import DTensor

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset
from nemo.collections.speechlm2.parts.optim_setup import freeze_and_subset
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# Gradient clipping is folded into the optimizer's fp32 divide rather than
# performed as a separate framework-native call, for the fp16/DTensor
# overflow reason described in the module docstring. This threshold is
# expressed in *unscaled* gradient-norm units (i.e. after dividing out the
# loss scale), so it is stable across however the dynamic loss scale
# happens to be calibrated at any given step.
GRADIENT_CLIP_VAL = 1000.0

# The perception-to-LLM bridge layer (a single linear projection from the
# frozen speech encoder's output space into the language model's embedding
# space) benefits from a higher learning rate than the rest of the
# trainable parameters when it is randomly initialized, since it must learn
# a much larger effective transformation than an already-pretrained layer
# fine-tuning incrementally. This is an empirically-motivated default, not
# a theoretically-derived optimum, and is fully overridable via
# `model.optimizer.bridge_lr`.
BRIDGE_PARAM_PATTERN = r"^perception\.proj\..+$"
BRIDGE_LR = 5e-4

# Dynamic loss-scale backoff/growth parameters, in the same style as
# `torch.amp.GradScaler` (backoff_factor=0.5, growth_factor=2.0). The
# growth interval and scale ceiling below were calibrated against this
# model's own observed gradient-magnitude history across multiple short
# validation runs before being used for full production training.
LOSS_SCALE_BACKOFF_FACTOR = 0.5
LOSS_SCALE_GROWTH_FACTOR = 2.0
LOSS_SCALE_GROWTH_INTERVAL = 2000
MIN_LOSS_SCALE = 128.0
MAX_LOSS_SCALE = 8192.0


class StableSALM(SALM):
    def configure_optimizers(self):
        # Splits the bridge layer into its own parameter group at
        # BRIDGE_LR; every other trainable parameter uses the base
        # model.cfg.optimizer.lr. Follows the same freeze_and_subset +
        # id(param)-based re-grouping pattern NeMo's own SALM optimizer
        # setup uses.
        trainable_ids = {id(p) for p in freeze_and_subset(
            self.named_parameters(),
            exclude_patterns=self.cfg.get("freeze_params", []),
            keep_patterns=self.cfg.get("prevent_freeze_params", []),
        )}
        bridge_pattern = re.compile(BRIDGE_PARAM_PATTERN)
        bridge_params, rest_params = [], []
        for name, p in self.named_parameters():
            if id(p) not in trainable_ids:
                continue
            (bridge_params if bridge_pattern.match(name) else rest_params).append(p)

        if not bridge_params:
            logging.warning(
                f"configure_optimizers: no trainable params matched BRIDGE_PARAM_PATTERN="
                f"{BRIDGE_PARAM_PATTERN!r} -- the bridge-layer LR split is inactive this run "
                f"(all trainable params will use model.cfg.optimizer.lr)."
            )
        bridge_lr = self.cfg.optimizer.get("bridge_lr", BRIDGE_LR)
        logging.info(
            f"configure_optimizers: {len(bridge_params)} bridge params @ lr={bridge_lr}, "
            f"{len(rest_params)} other trainable params @ lr={self.cfg.optimizer.lr} (default)."
        )

        optimizer_cfg = {k: v for k, v in self.cfg.optimizer.items() if k != "bridge_lr"}
        optim_groups = [
            {"params": bridge_params, "lr": bridge_lr},
            {"params": rest_params},
        ]
        optimizer = hydra.utils.instantiate(optimizer_cfg, optim_groups, _convert_="all")
        ans = {"optimizer": optimizer}
        if "lr_scheduler" in self.cfg:
            lr_scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler, optimizer)
            ans["lr_scheduler"] = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return ans

    def training_step(self, batch: dict, batch_idx: int):
        ans = super().training_step(batch, batch_idx)
        train_loss = ans["loss"].detach()
        if torch.distributed.get_rank() == 0 and batch_idx % 8 == 0:
            logging.info(f"training_step: batch_idx={batch_idx} train_loss_unscaled={train_loss.item():.6g}")
        # Only the returned (backward-bound) loss needs scaling. The live
        # scale is read off the optimizer, the single source of truth,
        # since on_before_optimizer_step adjusts it every step.
        ans = dict(ans)
        loss_scale = self.optimizers().loss_scale
        scaled_loss = ans["loss"] * loss_scale
        # The unscaled loss is fp16 (computed under precision: 16-true),
        # with max representable value 65504. At the calibrated loss-scale
        # ceiling, an unusually large unscaled loss can overflow this
        # multiply directly -- a distinct, upstream failure mode from the
        # gradient-based skip detected in on_before_optimizer_step, but one
        # that produces an outwardly identical "skipped" log line. Flagged
        # explicitly so the two are not confused when diagnosing a run.
        if torch.distributed.get_rank() == 0 and not torch.isfinite(scaled_loss):
            logging.warning(
                f"training_step: SCALED LOSS overflowed fp16 (loss={train_loss.item():.6g} * "
                f"loss_scale={loss_scale}) -- this step's skip (if any) is a loss-magnitude "
                f"overflow, not gradient instability."
            )
        ans["loss"] = scaled_loss
        return ans

    # configure_gradient_clipping is intentionally not overridden. The
    # framework's default implementation is a no-op when
    # trainer.gradient_clip_val is left unset, which this project's configs
    # do -- clipping happens entirely inside on_before_optimizer_step,
    # below, folded into the optimizer's existing fp32 divide.

    def on_before_optimizer_step(self, optimizer):
        # All per-parameter arithmetic below operates on each DTensor's
        # local shard only (zero collective operations), followed by
        # exactly one unconditional all-reduce per step. Branching on a
        # per-rank local value here (rather than the already-global
        # all-reduced result) risks different ranks taking different
        # control-flow paths and permanently desyncing the distributed
        # process group's expected collective-operation sequence.
        params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]

        local_bad = torch.zeros((), device=self.device, dtype=torch.float32)
        local_sq = torch.zeros((), device=self.device, dtype=torch.float32)
        for p in params:
            g = p.grad.detach()
            g = g.to_local() if isinstance(g, DTensor) else g
            gf = g.float()
            local_bad += (~torch.isfinite(gf)).any().to(torch.float32)
            local_sq += torch.nan_to_num(gf, nan=0.0, posinf=0.0, neginf=0.0).pow(2).sum()

        flags = torch.stack([local_bad, local_sq])
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.SUM)
        any_nonfinite = bool(flags[0].item() > 0.0)
        # Captured before any mutation below, since it is the scale that
        # was actually in effect when this step's backward() ran.
        loss_scale_used = optimizer.loss_scale
        # On a skipped step, flags[1] is contaminated by nan_to_num zeroing
        # out the non-finite contributions and is therefore not a
        # meaningful norm -- must not be logged or used as one.
        grad_norm_unscaled = float(flags[1].sqrt().item()) / loss_scale_used if not any_nonfinite else float("nan")

        clip_coef = float("nan")
        if any_nonfinite:
            for p in params:
                p.grad = None
            optimizer.loss_scale = max(optimizer.loss_scale * LOSS_SCALE_BACKOFF_FACTOR, MIN_LOSS_SCALE)
            optimizer._consecutive_clean_steps = 0
            optimizer.grad_divisor = optimizer.loss_scale

            optimizer._consecutive_skipped_steps = getattr(optimizer, "_consecutive_skipped_steps", 0) + 1
            if optimizer._consecutive_skipped_steps > 50:
                raise RuntimeError(
                    f"Sustained-skip tripwire: {optimizer._consecutive_skipped_steps} consecutive "
                    f"non-finite-gradient steps (loss_scale backed off to {optimizer.loss_scale}, "
                    f"floor {MIN_LOSS_SCALE}). This indicates the model has likely diverged rather "
                    "than hit a transient instability. Stopping rather than continuing to consume "
                    "compute on a diverged run."
                )
        else:
            optimizer._consecutive_skipped_steps = 0
            optimizer._consecutive_clean_steps = getattr(optimizer, "_consecutive_clean_steps", 0) + 1
            if optimizer._consecutive_clean_steps >= LOSS_SCALE_GROWTH_INTERVAL:
                optimizer.loss_scale = min(optimizer.loss_scale * LOSS_SCALE_GROWTH_FACTOR, MAX_LOSS_SCALE)
                optimizer._consecutive_clean_steps = 0

            # Equivalent to clipping the unscaled gradient to
            # GRADIENT_CLIP_VAL and then dividing by loss_scale_used, done
            # as a single fp32 operation inside the optimizer:
            # grad_divisor = loss_scale_used / clip_coef.
            clip_val = self.cfg.get("gradient_clip_val_unscaled", GRADIENT_CLIP_VAL)
            clip_coef = min(1.0, clip_val / (grad_norm_unscaled + 1e-12))
            optimizer.grad_divisor = loss_scale_used / clip_coef

            if not getattr(optimizer, "_verified_first_real_update", False):
                optimizer._pending_first_update_check = True

        if torch.distributed.get_rank() == 0:
            logging.info(
                f"on_before_optimizer_step: grad_norm_unscaled={grad_norm_unscaled:.6g} "
                f"loss_scale_used={loss_scale_used:.1f} loss_scale_next={optimizer.loss_scale:.1f} "
                f"clip_coef={clip_coef:.6g} clip_val={self.cfg.get('gradient_clip_val_unscaled', GRADIENT_CLIP_VAL):.1f} "
                f"skipped={any_nonfinite}"
            )

    def on_before_zero_grad(self, optimizer):
        # Fires after optimizer.step(), so a real update (if the previous
        # step was clean) has already happened by this point -- verify it
        # actually did, once, the first time a non-skipped step occurs.
        if getattr(optimizer, "_pending_first_update_check", False):
            optimizer._pending_first_update_check = False
            any_nonzero = any(
                torch.count_nonzero(
                    state["exp_avg"].to_local() if isinstance(state["exp_avg"], DTensor) else state["exp_avg"]
                ).item() > 0
                for state in optimizer.state.values()
                if "exp_avg" in state
            )
            if not any_nonzero:
                raise RuntimeError(
                    "Correctness tripwire: the first non-skipped optimizer step completed but "
                    "every trainable parameter's exp_avg is still exactly zero -- gradients are "
                    "being lost somewhere between on_before_optimizer_step and the optimizer's "
                    "own step(). Do not proceed."
                )
            optimizer._verified_first_real_update = True
            if torch.distributed.get_rank() == 0:
                logging.info("Correctness tripwire: PASSED -- confirmed a real (nonzero) optimizer update occurred.")

    def on_validation_epoch_end(self):
        # Writes every validation check's result unconditionally to a
        # JSONL file, so the full training/validation trajectory can be
        # reconstructed after the fact regardless of what the framework's
        # own console logging happens to print on any given check.
        super().on_validation_epoch_end()
        if self.trainer.sanity_checking or torch.distributed.get_rank() != 0:
            return
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return
        val_loss = float(val_loss)
        prev_best = getattr(self, "_best_val_loss_seen", float("inf"))
        is_new_best = val_loss < prev_best
        self._best_val_loss_seen = val_loss if is_new_best else prev_best
        row = {
            "wall_time": time.time(),
            "global_step": int(self.trainer.global_step),
            "epoch": int(self.trainer.current_epoch),
            "val_loss": val_loss,
            "best_val_loss_so_far": float(self._best_val_loss_seen),
            "is_new_best": is_new_best,
            "lrs": [g["lr"] for g in self.optimizers().param_groups] if self.trainer.optimizers else None,
        }
        log_path = getattr(self, "_metrics_log_path", "val_metrics.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        logging.info(f"on_validation_epoch_end: {row}")


@hydra_runner(config_path="conf", config_name="salm")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        model = StableSALM(OmegaConf.to_container(cfg.model, resolve=True))

    # Loads the released model's own trained bridge/encoder/decoder weights
    # on top of the freshly-composed architecture (see module docstring).
    if cfg.model.get("load_released_pretrained", False):
        from torch.distributed.tensor import DTensor, distribute_tensor

        has_local_lora = cfg.model.get("lora", None) is not None
        if torch.distributed.get_rank() == 0:
            logging.info("load_released_pretrained=True: loading the released model's "
                         "actual trained weights (bridge + encoder + LoRA-merged decoder) "
                         f"on top of the fresh composition. has_local_lora={has_local_lora}.")
        ref = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
        ref.llm = ref.llm.merge_and_unload()
        ref_state = ref.state_dict()
        del ref

        model_state = model.state_dict()

        if has_local_lora:
            # For a matched adaptation-scope comparison, the LoRA arm must
            # start from the exact same decoder weights as any full-decoder
            # arm (the merged, released, correctly-initialized decoder),
            # with its own LoRA adapters at standard PEFT cold-start
            # initialization (zero initial delta) rather than the released
            # model's own already-trained LoRA delta -- transferring that
            # would not be a fair "fresh adaptation" starting point.
            def _map_key(k: str) -> str:
                if not k.startswith("llm."):
                    return k
                rest = k[len("llm."):]
                candidate = f"llm.base_model.model.{rest}"
                if candidate in model_state:
                    return candidate
                for suffix in (".weight", ".bias"):
                    if rest.endswith(suffix):
                        candidate2 = f"llm.base_model.model.{rest[:-len(suffix)]}.base_layer{suffix}"
                        if candidate2 in model_state:
                            return candidate2
                raise KeyError(f"load_released_pretrained: could not map ref key {k!r} onto "
                                f"the local LoRA-wrapped model's state_dict -- LoRA config "
                                f"mismatch between the released model and this run's model.lora?")
            ref_state = {_map_key(k): v for k, v in ref_state.items()}

        converted = {}
        for k, v in ref_state.items():
            existing = model_state[k]
            if isinstance(existing, DTensor):
                converted[k] = distribute_tensor(v.to(existing.device_mesh.device_type), existing.device_mesh, existing.placements)
            else:
                converted[k] = v
        load_result = model.load_state_dict(converted, strict=False if has_local_lora else True)
        if torch.distributed.get_rank() == 0:
            logging.info(f"load_released_pretrained: loaded {len(converted)} tensors, "
                         f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}")
            if has_local_lora:
                n_missing = len(load_result.missing_keys)
                assert n_missing == 112, (
                    f"load_released_pretrained with LoRA: expected exactly 112 missing keys "
                    f"(the fresh-init lora_A/lora_B adapter params, left untouched by design), "
                    f"got {n_missing}. Investigate before trusting this run's initialization: "
                    f"missing={load_result.missing_keys}")
                assert len(load_result.unexpected_keys) == 0

    model._metrics_log_path = str(log_dir / "val_metrics.jsonl")

    dataset = SALMDataset(tokenizer=model.tokenizer)
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
