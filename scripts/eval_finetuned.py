#!/usr/bin/env python3
"""Evaluate a fine-tuned Canary-Qwen checkpoint (distributed FSDP checkpoint
format) via greedy decoding, reporting word error rate against a manifest
of reference transcriptions.

Usage: CUDA_VISIBLE_DEVICES=0 python eval_finetuned.py \
    --checkpoint <path>/checkpoints/step=N-last.ckpt \
    --test-manifest <path>/test_manifest.json
"""
import argparse, hashlib, json, os, traceback, torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from jiwer import wer

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test-manifest', required=True)
    p.add_argument('--output', default='eval_results.json')
    p.add_argument('--max-samples', type=int, default=0)
    # 'released' (default) is correct for a LoRA checkpoint trained on top
    # of the released base model -- LoRA-adapter-shaped parameter names
    # line up directly against that architecture. 'composed' instead
    # reconstructs the exact architecture a full-decoder (non-LoRA)
    # fine-tuning run was trained under, from that run's own saved
    # exp_config.yaml -- required whenever the checkpoint's parameter names
    # are plain (unwrapped) decoder weights rather than LoRA-adapter-shaped,
    # since loading those onto a LoRA-shaped base would silently mismatch
    # every language-model weight.
    p.add_argument('--base', choices=['released', 'composed'], default='released')
    p.add_argument('--exp-config', default=None,
                    help="Path to the run's exp_config.yaml (written next to its checkpoints "
                         "by the training script). Required when --base composed.")
    args = p.parse_args()
    if args.base == 'composed' and not args.exp_config:
        p.error('--exp-config is required when --base composed')

    # The consolidated-weights cache path is derived from the checkpoint's
    # own path (not a single fixed path), so that concurrent evaluations of
    # different checkpoints never collide and silently reuse the wrong
    # cached weights.
    ckpt_hash = hashlib.sha256(os.path.abspath(args.checkpoint).encode()).hexdigest()[:16]
    consolidated = f'/tmp/canary_eval_consolidated_{ckpt_hash}.pt'
    if not os.path.exists(consolidated):
        print(f'Consolidating {args.checkpoint} -> {consolidated}...')
        dcp_to_torch_save(args.checkpoint, consolidated)
    else:
        print(f'Reusing cached consolidation for this exact checkpoint: {consolidated}')

    state = torch.load(consolidated, map_location='cpu', weights_only=False)
    if 'state_dict' in state: state = state['state_dict']
    nans = sum(1 for v in state.values() if torch.isnan(v).any())
    print(f'NaN check: {nans}/{len(state)}')
    assert nans == 0, 'NaN weights detected — check AdamW eps setting'

    from nemo.collections.speechlm2.models import SALM
    print(f'Loading model (base={args.base})...')
    if args.base == 'released':
        model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
        base_desc = 'nvidia/canary-qwen-2.5b (released)'
    else:
        from omegaconf import OmegaConf
        exp_cfg = OmegaConf.load(args.exp_config)
        model = SALM(OmegaConf.to_container(exp_cfg.model, resolve=True))
        # Composed construction defaults to fp32; the checkpoint's own
        # tensors are fp16 (trained under fp16 precision), and fp32 for
        # this model's parameter count exceeds available GPU memory on
        # commodity hardware. Cast to match the checkpoint's dtype.
        model = model.half()
        base_desc = (f"composed from {args.exp_config} "
                     f"(pretrained_llm={exp_cfg.model.pretrained_llm}, "
                     f"pretrained_asr={exp_cfg.model.pretrained_asr})")

    # strict=False alone would silently discard any missing/unexpected
    # keys rather than surface them. Given the structural-mismatch risk
    # described above (LoRA-shaped vs. plain decoder parameter names), a
    # silent partial load could report the base model's own performance
    # rather than the checkpoint's -- so the actual mismatch counts are
    # asserted on explicitly, regardless of which --base was used.
    load_result = model.load_state_dict(state, strict=False)
    n_missing, n_unexpected = len(load_result.missing_keys), len(load_result.unexpected_keys)
    print(f'load_state_dict: {n_missing} missing keys, {n_unexpected} unexpected keys')
    if n_missing:
        print(f'  missing (first 10): {load_result.missing_keys[:10]}')
    if n_unexpected:
        print(f'  unexpected (first 10): {load_result.unexpected_keys[:10]}')
    # --base composed reconstructs the exact architecture the checkpoint
    # was trained under, so a healthy load has no excuse for any key
    # mismatch -- require exactly 0. --base released keeps a small
    # tolerance, since that path loads a checkpoint's weights onto an
    # independently-constructed base that can have a handful of harmless
    # buffer-naming differences.
    max_allowed_mismatch = 0 if args.base == 'composed' else 5
    assert n_missing + n_unexpected <= max_allowed_mismatch, (
        f'{n_missing} missing + {n_unexpected} unexpected keys when loading '
        f'{args.checkpoint} onto {base_desc} -- this looks like a structural '
        f'mismatch (e.g. LoRA-shaped base model vs a plain full-parameter checkpoint), not '
        f'a handful of harmless buffer differences. Evaluating anyway would silently report '
        f"the base model's performance, not this checkpoint's."
    )
    model.cuda().eval()

    samples = [json.loads(l) for l in open(args.test_manifest)]
    if args.max_samples > 0: samples = samples[:args.max_samples]
    print(f'Evaluating {len(samples)} samples...')

    # Not passing a generation_config does not by itself guarantee greedy
    # decoding: the underlying generation library backfills every
    # default-valued field from the language model's own
    # generation_config.json for whatever this call leaves unset. When the
    # LLM is loaded with its published pretrained weights (the --base
    # composed path), that file sets do_sample=True with a non-trivial
    # temperature/top_k/top_p by default -- silently turning an intended
    # greedy decode into sampling. Passing do_sample (and the other
    # sampling-related fields) as direct keyword arguments to generate(),
    # rather than nested inside a GenerationConfig object, applies after
    # that backfill and reliably yields deterministic greedy decoding —
    # verified by confirming byte-identical output across repeated runs
    # of the same samples.
    gen_kwargs = dict(do_sample=False, num_beams=1, temperature=None, top_k=None, top_p=None)

    refs, hyps, errors = [], [], 0
    for i, s in enumerate(samples):
        if (i+1) % 500 == 0:
            print(f'  {i+1}/{len(samples)} WER: {wer(refs, hyps)*100:.1f}%')
        try:
            ids = model.generate(prompts=[[{'role':'user',
                'content':f'Transcribe the following: {model.audio_locator_tag}',
                'audio':[s['audio_filepath']]}]], max_new_tokens=128,
                **gen_kwargs)
            refs.append(s['text'].lower().strip())
            hyps.append(model.tokenizer.ids_to_text(ids[0].cpu()).lower().strip())
        except Exception as e:
            # Reports the actual exception rather than silently counting
            # it, so an unexpectedly high error count on a small sample
            # set can be diagnosed instead of just observed.
            errors += 1
            print(f'  [error] sample {i} ({s.get("audio_filepath", "?")}): {type(e).__name__}: {e}')
            traceback.print_exc()

    final_wer = wer(refs, hyps)
    print(f'\nWER: {final_wer*100:.2f}% ({len(refs)} samples, {errors} errors)')
    json.dump({'wer': final_wer, 'samples': len(refs), 'errors': errors},
              open(args.output, 'w'), indent=2)

if __name__ == '__main__': main()
