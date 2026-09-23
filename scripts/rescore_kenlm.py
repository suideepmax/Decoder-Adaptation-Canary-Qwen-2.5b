#!/usr/bin/env python3
"""Rescore Canary-Qwen N-best hypotheses (from generate_nbest.py) with the same
in-domain KenLM binary used for the W2V2 baseline (S4-FAIR decoding-fairness
comparison). Must run in an environment with the `kenlm` python bindings
(this repo's `w2v2_asr` conda env; `canary_ft` does not have it installed).

Fusion: fused_score = model_score + alpha * (kenlm_log10_score * ln(10) / num_words)
  - model_score: HF beam-search sequence score (natural-log, length-normalized
    per token by transformers' default length_penalty=1.0).
  - kenlm_log10_score: kenlm.Model.score(text, bos=True, eos=True), total
    log10 probability of the whole hypothesis; converted to natural log and
    normalized per word so it is on a comparable per-unit scale to model_score.
  - alpha is swept over a small fixed grid rather than tuned on the test set
    (tuning fusion weight on the eval set would be optimistic); alpha=0.5
    is reported as the headline number because it matches pyctcdecode's own
    default alpha used for the W2V2+KenLM baseline (apples-to-apples choice,
    not selected by test-set performance).

Usage: conda activate w2v2_asr && python rescore_kenlm.py \
    --nbest nbest_v1_full.json \
    --kenlm ~/w2v2-air-traffic/experiments/data/uwb_atcc/train/lm/uwb_atcc_4g.binary \
    --output kenlm_rescore_results.json
"""
import argparse, json, math
import kenlm
from jiwer import wer

ALPHA_GRID = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
HEADLINE_ALPHA = 0.5

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nbest', required=True)
    p.add_argument('--kenlm', required=True)
    p.add_argument('--output', default='kenlm_rescore_results.json')
    args = p.parse_args()

    lm = kenlm.Model(args.kenlm)
    data = json.load(open(args.nbest))
    results = data['results']
    print(f'Loaded {len(results)} samples, {data["num_beams"]}-best each, {data["errors"]} generation errors excluded.')

    # Precompute per-hyp kenlm score once (independent of alpha).
    for r in results:
        for h in r['hyps']:
            n_words = max(1, len(h['text'].split()))
            log10_score = lm.score(h['text'], bos=True, eos=True)
            h['kenlm_per_word_nats'] = (log10_score * math.log(10)) / n_words

    refs = [r['text'] for r in results]
    native_hyps = [r['hyps'][0]['text'] for r in results]  # rank-1 beam, no LM (native comparison point)
    native_wer = wer(refs, native_hyps)
    print(f'Native (beam rank-1, no KenLM): WER = {native_wer*100:.2f}%')

    sweep = {}
    for alpha in ALPHA_GRID:
        rescored_hyps = []
        for r in results:
            best = max(r['hyps'], key=lambda h: h['model_score'] + alpha * h['kenlm_per_word_nats'])
            rescored_hyps.append(best['text'])
        w = wer(refs, rescored_hyps)
        sweep[alpha] = w
        print(f'alpha={alpha:.1f}: WER = {w*100:.2f}%')

    headline_hyps = []
    for r in results:
        best = max(r['hyps'], key=lambda h: h['model_score'] + HEADLINE_ALPHA * h['kenlm_per_word_nats'])
        headline_hyps.append(best['text'])
    headline_wer = wer(refs, headline_hyps)

    out = {
        'samples': len(results),
        'num_beams': data['num_beams'],
        'native_beam_rank1_wer': native_wer,
        'headline_alpha': HEADLINE_ALPHA,
        'headline_nbest_kenlm_wer': headline_wer,
        'alpha_sweep': {str(a): w for a, w in sweep.items()},
        'note': 'alpha=0.5 chosen to match pyctcdecode default used for W2V2+KenLM baseline, not tuned on this test set.',
    }
    json.dump(out, open(args.output, 'w'), indent=2)
    print(f'\nHeadline (alpha={HEADLINE_ALPHA}): N-best+KenLM WER = {headline_wer*100:.2f}%')
    print(f'Saved to {args.output}')

if __name__ == '__main__': main()
