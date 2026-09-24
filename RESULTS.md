# Full Results

Every measured result referenced in the paper, in full detail, organized
by experiment. All word error rates (WER) are computed with `jiwer` after
lowercasing and stripping whitespace from both reference and hypothesis
text. Unless stated otherwise, WER is reported on the full 2,886-utterance
UWB-ATCC test set.

## 1. External baseline: Wav2Vec2-large CTC

Model: `facebook/wav2vec2-large-960h-lv60-self` (317M parameters, full
fine-tuning), trained for 10,000 steps, effective batch size 64
(1 × 16 × 4 GPUs, DDP), learning rate 5e-4.

| Metric | This work | UWB-ATCC-only released model (HF card) | Joint UWB-ATCC+ATCOSIM released model (HF card) |
|---|---|---|---|
| WER, no LM (greedy) | **14.54%** | 17.56% | 17.48% |
| WER, with in-domain 4-gram KenLM | **12.69%** | 13.72% | 14.26% |

Learning-rate note: the released UWB-ATCC-only model was trained at
1e-4; this work used 5e-4 (5×), with a larger effective batch via DDP
across 4 GPUs, reaching lower WER in the same 10,000 steps.

Training curve (this work, 3 independent runs, training-time greedy
eval): 15.07% / 15.15% / 15.17% WER at step 10,000 (train loss
0.4062 / 0.4076 / 0.4077). Standalone final evaluation (greedy, no LM)
on the selected checkpoint: 14.54% (canonical), with a second standalone
eval of the same checkpoint giving 14.60% (0.06pp run-to-run variation
under otherwise identical conditions).

KenLM impact: −1.85 percentage points (14.54% → 12.69%).

## 2. Canary-Qwen original LoRA-scope study (v1 / v2 / v3)

Canary-Qwen-2.5B: frozen Canary-1B-Flash speech encoder, linear
connector, Qwen3-1.7B decoder. All three composed fresh from base
components (`Qwen/Qwen3-1.7B` + `nvidia/canary-1b-flash`), i.e. with a
randomly-initialized connector rather than the released model's own
trained connector (see Section 6). Trained under a plain fp16 AdamW
optimizer (see Section 8 for the numerical implications of this).
10,000 steps, effective batch 32, on the original 11,543-cut training
split (10.54h of audio).

| Config | Adaptation scope | Regularization | Test WER |
|---|---|---|---|
| v1 (`configs/v1_lora_baseline.yaml`) | LoRA r=128, q_proj+v_proj only (27.8M / 0.97% of params) | None (dropout=0.01, weight_decay=1e-3) | **23.32%** |
| v2 (`configs/v2_encoder_unfrozen.yaml`) | LoRA r=128 (as v1) + full encoder fine-tuning (838.8M params, 29.2% of model) | None | **23.82%** |
| v3 (`configs/v3_lora_regularized.yaml`) | LoRA r=128 (as v1) | SpecAugment (2 freq masks, 10 time masks) + LoRA dropout 0.1 + weight_decay=1e-2 | **20.70%** |

v1 result independently re-verified across three separate checks: a
fresh retraining of the identical configuration (23.3210%, matching to
four significant figures), and inference against the author-uploaded
HuggingFace checkpoint. v3's result (20.7004%) was independently
re-verified via a local checkpoint re-evaluation and inference against
the author-uploaded HuggingFace checkpoint, both bit-for-bit identical
to 16 significant figures.

Unfreezing the encoder alone (v2), without regularization, does not beat
LoRA-only adaptation (23.82% vs. 23.32%) — the ~24% plateau shared by v1
and v2 is attributable to overfitting on the training set at this scale,
not to the frozen decoder being an architectural bottleneck; v3 breaks
through to 20.70% via regularization alone, at the same learning rate,
same adaptation scope as v1.

Weight decay's role in v3's improvement is ruled out with certainty: see
Section 9 (regularization decomposition) and Section 8 (numerical
stability) — the plain fp16 optimizer used for v1/v2/v3 makes weight
decay numerically inert at every value used in this project, regardless
of its configured setting. v3's gain is attributable to SpecAugment
and/or increased LoRA dropout.

## 3. Decoding-fairness analysis (N-best + in-domain KenLM)

For every checkpoint below: 5-best beam search generation, then
rescoring with the same in-domain 4-gram KenLM used for the Wav2Vec2
baseline (Section 1). Rescoring weight (α) swept over {0.0, 0.1, 0.3,
0.5, 0.7, 1.0}; α=0.5 is the pre-registered headline value (matching the
default fusion weight used for the Wav2Vec2+KenLM baseline, not tuned on
this test set).

### v1

| Decoding | WER |
|---|---|
| Native, greedy | 23.32% |
| 5-beam, rank-1, no LM | 22.28% |
| 5-best + KenLM, α=0.5 (headline) | 21.79% |
| 5-best + KenLM, best α (0.3) | 21.71% |

Beam search alone accounts for roughly two-thirds of the total 1.53-point
improvement from native greedy to the best rescored result; the external
LM adds the remaining third.

### v3

| Decoding | WER |
|---|---|
| Native, greedy | 20.70% |
| 5-beam, rank-1, no LM | 19.42% |
| 5-best + KenLM, α=0.5 (headline) | 20.14% (worse than beam alone) |
| 5-best + KenLM, best α (0.1) | 19.44% (statistical tie with beam alone) |

Unlike v1, the external LM provides no benefit for v3 at any tested α up
to the headline value, and is actively harmful at the headline α=0.5. All
of v3's improvement over greedy decoding comes from beam search itself.

### Matched-protocol full-decoder arm (Section 5)

| Decoding | WER |
|---|---|
| Native, greedy | 18.73% |
| 5-beam, rank-1, no LM | 17.54% |
| 5-best + KenLM, α=0.5 (headline) | 18.66% (worse than beam alone) |
| 5-best + KenLM, best α (0.1) | 17.67% |

### Matched-protocol LoRA arm, full exposure (Section 5)

| Decoding | WER |
|---|---|
| Native, greedy | 19.70% |
| 5-beam, rank-1, no LM | 18.67% |
| 5-best + KenLM, α=0.5 (headline) | 19.16% (worse than beam alone) |
| 5-best + KenLM, best α (0.1) | 18.59% |

Both matched-protocol arms show the same pattern as v3, not v1: beam
search alone gives essentially all of the available improvement, and the
external in-domain KenLM is neutral-to-harmful at the headline weight.
This is consistent with the hypothesis that well-regularized models
(SpecAugment + effective weight decay, present in all four of these
configurations but not in v1) are already well-calibrated to in-domain
phrasing, leaving little room for an external n-gram language model to
add value. Giving the SALM models the same language-model access as the
Wav2Vec2 baseline narrows, but does not close, the remaining gap to it.

## 4. Dataset integrity audits

**Development-set provenance.** The 915-utterance development set used
for checkpoint selection in the matched-protocol comparison (Section 5)
is a session-grouped subset carved out of the *original* 11,543-cut
training split (see `scripts/make_dev_split.py`). Direct utterance-ID
intersection confirms: 915/915 (100%) of the development set's
utterances were part of v1/v2/v3's own training data. It has zero
overlap with the corrected 10,619-cut training split used for the
matched-protocol arms. Practically: the development set is a valid
held-out set only for models trained on the corrected split; it must
never be used to compare v1/v2/v3 against the matched-protocol arms, since
v1/v2/v3 were trained directly on it. Every headline comparison in this
work between the two model families uses only the test set, which is
disjoint from both training splits.

**Train/test leakage in the original split.** One session
(`uwb-atcc_ACCU-pwnH5N`, 9 utterances) was present in both the original
11,543-cut training split and the 2,886-utterance test set. This session
is excluded when constructing the corrected training split.

**External baseline (Wav2Vec2) train/test check.** The Wav2Vec2 baseline
uses a separate, Kaldi-format data pipeline (`wav.scp`/`segments`/`text`),
independently checked for the same class of leakage: zero exact-utterance
overlap between its 11,543-utterance train and 2,886-utterance test
splits. One session-level overlap was found (the same
`uwb-atcc_ACCU-pwnH5N` session — 9 segments in train, one different,
non-overlapping-time-range segment in test), affecting 1 of 2,886 test
utterances (0.035%). This is the same underlying source-corpus issue
surfacing independently in both data pipelines; its effect on the
reported WER is negligible (a single utterance cannot meaningfully move
an aggregate WER computed over 2,886 utterances) but is disclosed for
completeness.

## 5. Matched-protocol comparison

Full-decoder fine-tuning vs. LoRA-only adaptation, under an
initialization, optimizer, data split, and regularization scheme held
identical between arms — the only intended difference is adaptation
scope. See `README.md` and `configs/matched_*.yaml` for the full design
rationale. All three configurations load the released model's own
trained connector and encoder weights (`load_released_pretrained`, see
Section 6), use `master_weight_adamw.MasterWeightAdamW` (Section 8), and
train on the corrected 10,619-cut split.

### Learning-rate selection

Three candidate learning rates per arm, 375 steps each, evaluated by WER
on the full 915-sample development set:

| Arm | Candidate LR | val_loss @ step 375 | Dev WER |
|---|---|---|---|
| Full-decoder | 5e-6 | 0.785 | 31.17%* |
| Full-decoder | 1e-5 | 0.678 | 28.15%* |
| Full-decoder | **2e-5 (selected)** | **0.598** | **25.62%** |
| LoRA | 1e-4 | 0.942 | 37.66% |
| LoRA | 3e-4 | 0.765 | 30.01% |
| LoRA | **5e-4 (selected)** | **0.700** | **30.04%** |

*Evaluated on a 500-sample development subset rather than the full 915,
for these two full-decoder candidates only; not re-run on the full set
since the selection outcome (2e-5) is unambiguous by a wide margin on
validation loss alone. LoRA's 3e-4 vs. 5e-4 WER is statistically
indistinguishable at n=915 (0.03pp apart); 5e-4 was selected for its
still-improving validation loss with no plateau observed within the
probe's step budget.

### Full-decoder arm (`configs/matched_full_decoder.yaml`)

9,200 steps = 27.72 true epochs on the corrected data split.

| Checkpoint (step) | True epochs | Dev WER (n=915) |
|---|---|---|
| 2300 | 6.93 | 18.80% |
| 4600 | 13.86 | 18.32% |
| 6900 | 20.79 | 18.87% |
| **9200 (selected)** | **27.72** | **18.19%** |

**Test WER (n=2,886) for the selected checkpoint: 18.73%.**

Checkpoint-to-checkpoint WER is non-monotonic despite validation loss
reaching its minimum at step 750 (val_loss 0.562) and drifting upward
thereafter (plateauing in the 0.56-0.65 band through step 9,200) — see
Section 7.

### LoRA arm, truncated exposure (`configs/matched_lora_truncated.yaml`)

3,700 steps = 11.15 true epochs, a compute-budget-truncated run targeting
the exposure range where the comparison is most informative given a
fixed compute budget (LoRA's behavior at the full 27.72-epoch endpoint
was already characterized by v1/v3, Section 2).

| Checkpoint (step) | True epochs | Dev WER (n=915) |
|---|---|---|
| 925 | 2.79 | 25.20% |
| 1850 | 5.58 | 20.59% |
| 2775 | 8.36 | 20.00% |
| **3700 (selected)** | **11.15** | **19.83%** |

**Test WER (n=2,886) for the selected checkpoint: 20.62%.**

### LoRA arm, full exposure (`configs/matched_lora_full9200.yaml`)

9,200 steps = 27.72 true epochs, exactly matching the full-decoder arm's
exposure, with checkpoints aligned to the same step numbers where
possible (2300/4600/6900/9200 shared between both arms).

| Checkpoint (step) | True epochs | Dev WER (n=915) |
|---|---|---|
| 1150 | 3.47 | 22.07% |
| 2300 | 6.93 | 22.10% |
| 3450 | 10.40 | 22.29% |
| 4600 | 13.86 | 20.42% |
| 5750 | 17.33 | 18.96% |
| 6900 | 20.79 | 19.48% |
| 8050 | 24.26 | 17.90% |
| **9200 (selected)** | **27.72** | **17.83%** |

**Test WER (n=2,886) for the selected checkpoint: 19.70%.**

### Headline comparison, all three arms

| Arm | True epochs | Test WER |
|---|---|---|
| Full-decoder | 27.72 | **18.73%** |
| LoRA, truncated exposure | 11.15 | 20.62% |
| LoRA, full exposure | 27.72 | **19.70%** |

At fully matched exposure (both arms 27.72 true epochs), full-decoder
fine-tuning beats LoRA-only adaptation by **0.97 percentage points**
(18.73% vs. 19.70%). The truncated-exposure comparison alone (18.73% vs.
20.62%, a 1.89-point gap) somewhat overstates this advantage, since it
compares the full-decoder arm at its full exposure against LoRA at less
than half that exposure — the fully matched comparison is the more
defensible headline number. Both truncated- and full-exposure LoRA
results are reported, since the truncated run's exposure range is itself
informative about how quickly each adaptation scope's WER improves
relative to training exposure.

## 6. Effect of correct connector initialization (prior to the matched-protocol comparison)

Composing a SALM from its published base components by default gives a
randomly-initialized connector layer, not the released model's own
further-trained one (see `README.md` and `scripts/train_salm.py` for the
technical detail and the documented upstream configuration pitfall this
corrects). Before committing to the full matched-protocol runs, a
shorter diagnostic run under the corrected initialization (SpecAugment
added, otherwise matching the corrected pipeline but at reduced step
count) was evaluated at four checkpoints:

| Checkpoint (step) | True epochs | Dev WER (n=915) | Test WER (n=2,886) |
|---|---|---|---|
| 750 | 2.26 | 24.11% | 23.89% |
| 875 | 2.64 | 23.62% | 22.83% |
| 1125 | 3.40 | 23.69% | **22.49%** |
| 2000 (dev-selected) | 6.03 | **23.52%** | 22.88% |

This is a direct demonstration of the checkpoint-selection instability
discussed in Section 7: the checkpoint selected by development-set WER
(step 2000) is not the checkpoint with the best test-set WER (step
1125). Every one of these four checkpoints beats v1's canonical 23.32%
outright, at 6.03 true epochs or fewer (roughly a fifth or less of v1's
27.72-epoch training exposure) — the best (22.49% at 3.40 epochs) is
within 1.79 points of v3's 20.70% result at less than an eighth of v3's
training exposure. This result motivated committing compute to the full
matched-protocol comparison in Section 5.

Separately, an epoch-normalized comparison across all available
full-decoder and LoRA checkpoints found that comparing these two
training families by raw optimizer-step count is invalid: v1/v2/v3's
10,000 steps on the original 11,543-cut split correspond to 27.72 true
epochs, while early full-decoder experiments were run for step counts
that, on the corrected (differently-sized) data split, corresponded to
substantially fewer true epochs — a confound larger than initialization,
optimizer, or data-split differences combined, and the direct motivation
for reporting every result in this work in true-epoch terms rather than
raw step counts.

## 7. Checkpoint-selection instability

NeMo's default `ModelCheckpoint(monitor="val_loss")` behavior — saving or
selecting the checkpoint with the lowest validation loss — does not
reliably select the checkpoint with the best word error rate in this
training regime, demonstrated three separate ways:

1. **Full-decoder arm's own checkpoints are non-monotonic in WER**
   despite validation loss reaching its minimum early and drifting
   upward afterward (Section 5): the checkpoint with the best validation
   loss (step 750, val_loss 0.562, in the diagnostic run) does not have
   the best WER among that run's checkpoints (step 1125 does, Section 6).
2. **Development-set-selected checkpoint ≠ test-set-best checkpoint.**
   In the diagnostic run (Section 6), the development-set WER selects
   step 2000, but the test-set WER shows step 1125 is actually better
   (22.49% vs. 22.88%) — a legitimate dev/test mismatch (different splits
   sampling different utterances), not an error, but one with direct
   practical consequences for which checkpoint would be reported.
3. **Evaluation-sample-size sensitivity.** A 500-sample subset of the
   development set, the full 915-sample development set, and the full
   2,886-sample test set each identified a *different* checkpoint as
   "best" among the same four diagnostic-run checkpoints in Section 6.

Practical implication: any pipeline that saves only the single checkpoint
with the lowest framework-reported validation loss risks silently
discarding a better-performing checkpoint, with no indication in the
training log that this has happened. Every headline result in this work
is instead selected by evaluating every saved checkpoint's WER on the
full development set, with the test set touched exactly once per arm for
its development-set-selected winner.

## 8. Numerical stability under fp16 training

Under plain fp16 `torch.optim.AdamW` (used for v1/v2/v3, Section 2), the
second-moment estimate (`exp_avg_sq`) underflows to exactly zero at this
model's realistic gradient magnitudes, degenerating the AdamW update
into sign-based SGD-with-momentum at an effective learning rate of
`lr/eps`, not the intended adaptive update. This also makes weight decay
numerically inert: the decoupled-decay factor `1 - lr·weight_decay`
rounds to exactly 1.0 in fp16 whenever `lr·weight_decay` is below
roughly 4.9×10⁻⁴ — true for every learning-rate/weight-decay combination
used in this project (verified directly: `1 - 5e-4×1e-3` rounds to
exactly 1.0 in fp16 arithmetic). This is proven analytically and
verified numerically, not inferred from an ablation — see Section 9.

`scripts/master_weight_adamw.py` (`MasterWeightAdamW`, used for every
matched-protocol configuration and every result in Sections 5-7) keeps
fp32 master copies of each parameter and its optimizer state, performing
the full AdamW update in fp32 regardless of the model's fp16 storage
dtype, resolving this. `scripts/train_salm.py` additionally implements a
correctness fix for gradient-norm computation under FSDP2's sharded
(DTensor) parameters (squaring a sharded gradient's local shard in fp16
before the cross-rank all-reduce can overflow to `inf` at this model's
realistic gradient magnitudes, silently zeroing every gradient on that
step) and a determinism fix for greedy/beam-search decoding under the
released model's own generation defaults (see the docstrings in both
files for full technical detail, including the specific numerical checks
used to verify each fix).

## 9. Regularization decomposition

v3's three simultaneous changes over v1 (SpecAugment, LoRA dropout
0.01→0.1, weight_decay 1e-3→1e-2) were decomposed to identify which
change(s) drove the 23.32%→20.70% improvement.

**Weight decay is ruled out with certainty** (Section 8): numerically
inert in the optimizer used for this comparison, regardless of its
configured value.

**SpecAugment vs. LoRA dropout** were isolated via two additional
configurations (`configs/regularization_ablation_specaugment_only.yaml`,
`configs/regularization_ablation_dropout_only.yaml`), each an exact
minimal diff from v1 changing only the one variable being isolated,
trained with the same original (unmodified) training entrypoint used for
v1/v2/v3. Due to compute constraints, both ablations were run for only
500 steps (5% of v1/v3's original 10,000-step exposure) — a directional
result, not a definitive decomposition at matched exposure.

| Configuration | Test WER @ step 500 |
|---|---|
| v1 (no SpecAugment, dropout=0.01) | 39.65% |
| SpecAugment only (dropout unchanged at 0.01) | 41.41% (+1.76pp vs. v1) |
| LoRA dropout only (0.1, no SpecAugment) | 40.11% (+0.46pp vs. v1) |

The v1 reference point at step 500 was obtained at zero
additional compute cost, from an existing checkpoint of an earlier,
independent fresh retraining of the v1 configuration (used originally to
verify v1's result's reproducibility, Section 2), confirmed via its own
embedded training configuration to be genuine: LoRA dropout 0.01, weight
decay 0.001, no SpecAugment, at global step 500.

Both regularization components individually *increase* WER relative to
the unregularized baseline at this short exposure, consistent with the
well-documented tendency of these techniques to slow early convergence
while their generalization benefit emerges only over substantially
longer training. **Which specific component (or combination) drove v3's
actual, full-exposure improvement remains an open question** — the
definitive version of this experiment (both ablations at the full
10,000-step exposure, requiring approximately 42 GPU-hours total) is a
well-defined, not-yet-completed follow-up.
