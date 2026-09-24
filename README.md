# Decoder Adaptation Scope for Canary-Qwen-2.5B on Air Traffic Control Speech

A controlled study of decoder-adaptation scope for Canary-Qwen-2.5B, a
speech-LLM (SALM: speech encoder + language-model decoder) architecture,
on air traffic control (ATC) speech recognition. We compare parameter-
efficient (LoRA) and full-parameter decoder fine-tuning under a matched
experimental protocol, evaluate the effect of a documented model-loading
pitfall on the resulting comparison, and report a decoding-fairness
analysis against an external CTC baseline.

**[RESULTS.md](RESULTS.md) contains every measured result referenced in
the paper**, in full detail: the external Wav2Vec2 baseline's full
training curve, the original v1/v2/v3 study, decoding-fairness tables for
all four evaluated checkpoints (not just the two matched-protocol arms),
every per-checkpoint development-set WER for all three matched-protocol
arms, the learning-rate selection probes, the dataset-integrity audits,
and the full regularization-decomposition ablation. This README
summarizes the headline findings; RESULTS.md is the source of record for
every number.

## Summary

Two easily-missed defaults materially change the outcome of a decoder-
adaptation-scope comparison on this model:

1. **Connector (bridge) initialization.** Composing a SALM from its
   published base components (a frozen speech encoder plus a language
   model) by default gives a *randomly initialized* connector layer
   between them, rather than the released model's own further-trained
   connector weights. This is a documented configuration pitfall in the
   underlying framework, not something specific to this project, but it
   is easy to trigger unintentionally and changes every downstream
   result.
2. **Training exposure normalization.** Comparing two training runs by
   raw optimizer-step count is not meaningful when they use data splits
   of different total sizes. True data-epoch count, not step count, is
   the correct basis for matching two arms' training exposure.

Once both are corrected, a controlled three-arm comparison (full-decoder
fine-tuning, LoRA at truncated exposure, LoRA at fully matched exposure)
finds that full-decoder fine-tuning outperforms LoRA-only adaptation by
approximately one point of word error rate (WER) at matched training
exposure — narrowing, but not closing, the gap to an external CTC
baseline. We additionally show that this toolkit's default validation-
loss-based checkpoint selection can pick a suboptimal checkpoint,
demonstrated three separate ways at increasing evaluation-sample sizes.

## Dataset

[UWB-ATCC](https://huggingface.co/Jzuluaga/uwb_atcc) (University of West
Bohemia Air Traffic Control Communication corpus): manually transcribed
English ATC speech, standard tower/ground/approach phraseology, including
non-native speakers.

| Split | Cuts | Notes |
|---|---|---|
| Original train (v1/v2/v3 baselines) | 11,543 | Contains one session (9 utterances) that also appears in the test set — identified and excluded when the corrected split below was built. |
| Corrected train (matched-protocol arms) | 10,619 | Session-grouped exclusion of both the above leak and the held-out development sessions below. |
| Development (matched-protocol arms only) | 915 | Carved from the original training sessions, session-grouped so no session's utterances split across train/dev. **Not held-out relative to the original v1/v2/v3 baselines** — every development-set session was part of their training data. Held-out only relative to models trained on the corrected training split. |
| Test | 2,886 | Session-disjoint from all of the above; used for every headline result in this work. |

`scripts/make_dev_split.py` reproduces the corrected training/development
split from the original training manifest. Full dataset-integrity audit
results (leakage checks on both this pipeline and the external Wav2Vec2
baseline's independent data pipeline) are in [RESULTS.md](RESULTS.md) §4.

## Models compared

All Canary-Qwen models are Canary-Qwen-2.5B: a frozen Canary-1B-Flash
speech encoder, a linear connector layer, and a Qwen3-1.7B language-model
decoder. The Wav2Vec2 model is an independent, non-SALM external
baseline used throughout as the point of comparison.

| Model | Adaptation scope | Regularization | Initialization | Result (test WER, no LM) |
|---|---|---|---|---|
| `facebook/wav2vec2-large-960h-lv60-self` (external CTC baseline) | Full fine-tuning (317M params) | — | Released checkpoint | 14.54% (12.69% with in-domain KenLM) |
| `configs/v1_lora_baseline.yaml` | LoRA (r=128, q/v projections only) | None | Composed fresh (random connector) | 23.32% |
| `configs/v2_encoder_unfrozen.yaml` | LoRA + full encoder fine-tuning | None | Composed fresh | 23.82% |
| `configs/v3_lora_regularized.yaml` | LoRA (r=128, q/v) | SpecAugment + dropout 0.1 + weight decay | Composed fresh | 20.70% |
| `configs/matched_full_decoder.yaml` | Full decoder fine-tuning (no LoRA) | SpecAugment + weight decay | Released connector loaded | **18.73%** |
| `configs/matched_lora_truncated.yaml` | LoRA (r=128, q/v) | SpecAugment + dropout + weight decay | Released connector loaded | 20.62% (11.15 true epochs) |
| `configs/matched_lora_full9200.yaml` | LoRA (r=128, q/v) | SpecAugment + dropout + weight decay | Released connector loaded | 19.70% (27.72 true epochs, matched) |

v1/v2/v3 are the original baseline configurations; the `matched_*` configs
are the corrected-initialization, matched-protocol comparison this work's
main result is drawn from. See [RESULTS.md](RESULTS.md) §1-2 for the
Wav2Vec2 and v1/v2/v3 full training curves, and §5 for every
matched-protocol arm's complete per-checkpoint table.

## Method

### The matched-protocol comparison

`matched_full_decoder.yaml` and `matched_lora_full9200.yaml` are held
identical in every respect except adaptation scope: initialization,
optimizer, data split, regularization (SpecAugment, weight decay), and
total training exposure (9,200 optimizer steps = 27.72 true epochs on
the corrected data split, matching the original baseline configurations'
own 27.72-true-epoch exposure at their own, different, step-rate). Both
arms' learning rates were independently selected via a short probe
(three candidates each, evaluated on the full development set) rather
than sharing a single inherited value.

At matched exposure, full-decoder fine-tuning (18.73% test WER) beats
LoRA-only adaptation (19.70%) by **0.97 percentage points**. A third,
compute-budget-truncated LoRA run (`matched_lora_truncated.yaml`, 3,700
steps / 11.15 true epochs, targeting the exposure range where the
comparison is most informative given a fixed compute budget) reaches
20.62% — the truncated comparison alone would suggest a larger, 1.89-point
gap, which the fully matched run shows is partly an exposure-mismatch
artifact rather than a difference attributable to adaptation scope alone.

### Learning-rate selection

Each matched-protocol arm's learning rate was chosen from three
candidates via a short probe (375 steps each, evaluated by development-set
WER): full-decoder settled on 2e-5 (31.17% → 28.15% → 25.62% dev WER
across the three candidates), LoRA settled on 5e-4 (37.66% → 30.01% →
30.04%). Full probe results, including validation loss at each candidate,
are in [RESULTS.md](RESULTS.md) §5.

### Checkpoint-selection instability

Across all three matched-protocol arms, the checkpoint with the best
validation loss is not reliably the checkpoint with the best word error
rate. For the full-decoder arm, the four saved checkpoints' WER is
non-monotonic (18.80% → 18.32% → 18.87% → 18.19%) despite validation
loss reaching its minimum at the earliest of the four. We further show
that ranking checkpoints by WER on a 500-sample evaluation subset, the
full 915-sample development set, and the full 2,886-sample test set can
each identify a *different* checkpoint as "best." This has a direct
practical implication: any pipeline that saves only the checkpoint with
the lowest framework-reported validation loss (this toolkit's default
behavior) risks discarding a better-performing checkpoint without any
indication that this has happened.

### Decoding-fairness analysis

To check whether the comparison against the external CTC baseline (which
uses beam search with an in-domain KenLM language model) is fair to the
SALM models, `scripts/generate_nbest.py` and `scripts/rescore_kenlm.py`
reproduce the same beam-search-plus-KenLM decoding for the SALM
checkpoints, using the same in-domain 4-gram language model as the
baseline.

| Config | Native (greedy) | Beam search (rank-1, no LM) | Beam + in-domain KenLM (α=0.5) |
|---|---|---|---|
| v1 (original, unregularized) | 23.32% | 22.28% | 21.79% |
| v3 (original, regularized) | 20.70% | 19.42% | 20.14% (worse than beam alone) |
| Full-decoder (matched) | 18.73% | 17.54% | 18.66% (worse than beam alone) |
| LoRA (matched, full exposure) | 19.70% | 18.67% | 19.16% (worse than beam alone) |

For v1, the external LM still helps (beam+LM beats beam alone). For every
regularized configuration (v3 and both matched-protocol arms), beam
search alone accounts for essentially all of the available improvement,
and the external language model provides no further benefit at the
pre-registered rescoring weight — mildly harmful in all three cases. This
suggests that the same regularization that improves these models'
unassisted WER also makes them less receptive to external LM correction.
Full alpha-sweep tables (all 6 tested values per checkpoint) are in
[RESULTS.md](RESULTS.md) §3.

### Regularization decomposition

The regularization added between v1 and v3 (SpecAugment, increased LoRA
dropout, increased weight decay) was applied as three simultaneous
changes. Weight decay can be ruled out as a contributing factor with
certainty: under the plain fp16 `torch.optim.AdamW` optimizer used for
v1/v2/v3 (before the fp32-master-weight fix described in Numerical
stability notes below), the decoupled weight-decay update term is
numerically zero at every weight-decay value used in this project
(verified directly: the update factor rounds to exactly 1.0 in fp16
arithmetic at these magnitudes), so it could not have influenced training
regardless of its configured value.

`configs/regularization_ablation_specaugment_only.yaml` and
`configs/regularization_ablation_dropout_only.yaml` isolate the
remaining two changes from each other. Due to compute constraints, these
ablations were run for only 500 steps (5% of v1/v3's original 10,000-step
exposure) and should be read as directional, not definitive: at this
short exposure, both components individually *increase* WER relative to
the unregularized baseline (SpecAugment: +1.76 points; dropout: +0.46
points), consistent with the well-documented tendency of these
regularization techniques to slow early convergence while their
generalization benefit emerges only over substantially longer training.
**Which specific component drove v3's actual, full-exposure improvement
remains an open question** — the definitive version of this experiment
(both ablations at the full 10,000-step exposure) is a well-defined
follow-up not completed in this work.

## Scope

This work addresses only the UWB-ATCC decoder-adaptation-scope
comparison for Canary-Qwen-2.5B. It does not cover ATCOSIM (a separate,
higher-audio-quality ATC corpus on which a differently-scoped Canary-Qwen
LoRA study exists but is out of scope here) or any Wav2Vec2 result beyond
the single UWB-ATCC baseline used for the decoding-fairness comparison in
this work.

## Reproducing

```
torchrun --nproc_per_node=4 scripts/train_salm.py \
    --config-path=configs --config-name=matched_full_decoder

torchrun --nproc_per_node=4 scripts/train_salm.py \
    --config-path=configs --config-name=matched_lora_full9200

python scripts/eval_finetuned.py \
    --base composed --exp-config <run_dir>/exp_config.yaml \
    --checkpoint <run_dir>/checkpoints/step=N-last.ckpt \
    --test-manifest <path>/test_manifest.json
```

`v1_lora_baseline.yaml`, `v2_encoder_unfrozen.yaml`,
`v3_lora_regularized.yaml`, and the two regularization-ablation configs
were trained with NeMo's own standard `examples/speechlm2/salm_train.py`
entrypoint (unmodified), not `scripts/train_salm.py` — this matters for
exact reproduction, since `train_salm.py` additionally applies the
numerical-stability fixes described below, which were not present when
those five configurations were originally trained.

Decoding-fairness rescoring requires a `kenlm`-bindings environment
separate from the main training environment:

```
python scripts/generate_nbest.py --base composed \
    --exp-config <run_dir>/exp_config.yaml \
    --checkpoint <run_dir>/checkpoints/step=N-last.ckpt \
    --test-manifest <path>/test_manifest.json \
    --output nbest.json --num-beams 5

python scripts/rescore_kenlm.py --nbest nbest.json \
    --kenlm <path-to-4gram-arpa-or-binary> --output rescored.json
```

## Numerical stability notes

`scripts/train_salm.py` and `scripts/master_weight_adamw.py` implement
and document a numerical-stability fix required for training this model
under fp16 precision: plain fp16 AdamW underflows its second-moment
estimate to zero at this model's realistic gradient magnitudes,
degenerating the optimizer into sign-based SGD and, separately, making
weight decay numerically inert (see the Regularization decomposition
section above). `master_weight_adamw.py` keeps fp32 master weights and
optimizer state to avoid this. `train_salm.py`'s docstring documents this
and two further correctness fixes (gradient-norm computation under
sharded/distributed parameters, and generation-config determinism for
decoding) in full technical detail, verified via direct numerical checks
described inline. Exact hardware, software versions, random seed, and WER
computation methodology are in [RESULTS.md](RESULTS.md) §10.

## Limitations

- **Data-scale sensitivity is untested.** All comparisons in this work
  use a fixed training-set size; how the full-decoder-vs-LoRA gap
  behaves as available in-domain data grows or shrinks is not addressed
  here.
- **Encoder adaptation under corrected initialization is untested.** The
  speech encoder is frozen in every matched-protocol configuration in
  this work. An earlier, differently-configured encoder-unfrozen
  baseline (v2) exists but predates the connector-initialization
  correction and the matched-protocol regularization scheme, so it
  cannot be used to answer this question cleanly.
- **Single random seed** throughout.
- **The regularization decomposition is directional only** (500-step
  exposure vs. the original 10,000-step comparison it addresses), as
  discussed above.
- **Checkpoint-to-checkpoint WER is non-monotonic** within every
  matched-protocol arm; the reported headline numbers are the
  development-set-selected best checkpoint per arm, with the full
  per-checkpoint table available for inspection, not a cherry-picked
  single number.
