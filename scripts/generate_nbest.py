#!/usr/bin/env python3
"""Generate N-best beam-search hypotheses from a fine-tuned Canary-Qwen
checkpoint, for the decoding-fairness comparison against an
externally-rescored baseline: produces N-best hypotheses with
per-sequence model scores so a separate script (rescore_kenlm.py, run in
an environment with `kenlm` Python bindings) can rescore them with the
same in-domain n-gram language model used for the comparison baseline.

Usage: CUDA_VISIBLE_DEVICES=0 python generate_nbest.py \
    --checkpoint <path>/checkpoints/step=N-last.ckpt \
    --test-manifest <path>/test_manifest.json \
    --output nbest_results.json --num-beams 5
"""
import argparse, hashlib, json, os, torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test-manifest', required=True)
    p.add_argument('--output', default='nbest_results.json')
    p.add_argument('--num-beams', type=int, default=5)
    p.add_argument('--max-samples', type=int, default=0)
    # 'released' is correct for a LoRA checkpoint trained on top of the
    # released base model. 'composed' reconstructs the exact architecture
    # a full-decoder (non-LoRA) fine-tuning run was trained under, from
    # that run's own saved exp_config.yaml -- required whenever the
    # checkpoint's parameter names are plain decoder weights rather than
    # LoRA-adapter-shaped, or load_state_dict(strict=False) will silently
    # discard most of the trained weights instead of erroring.
    p.add_argument('--base', choices=['released', 'composed'], default='released')
    p.add_argument('--exp-config', default=None,
                    help="Required when --base composed.")
    args = p.parse_args()
    if args.base == 'composed' and not args.exp_config:
        p.error('--exp-config is required when --base composed')

    # Cache path derived from the checkpoint's own path, avoiding
    # collisions between concurrent evaluations of different checkpoints.
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
    assert nans == 0, 'NaN weights detected'

    from nemo.collections.speechlm2.models import SALM
    print(f'Loading model (base={args.base})...')
    if args.base == 'released':
        model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
        base_desc = 'nvidia/canary-qwen-2.5b (released)'
    else:
        from omegaconf import OmegaConf
        exp_cfg = OmegaConf.load(args.exp_config)
        model = SALM(OmegaConf.to_container(exp_cfg.model, resolve=True))
        # Composed construction defaults to fp32; checkpoint tensors are
        # fp16 -- cast to match, same as eval_finetuned.py.
        model = model.half()
        base_desc = (f"composed from {args.exp_config} "
                     f"(pretrained_llm={exp_cfg.model.pretrained_llm}, "
                     f"pretrained_asr={exp_cfg.model.pretrained_asr})")

    load_result = model.load_state_dict(state, strict=False)
    n_missing, n_unexpected = len(load_result.missing_keys), len(load_result.unexpected_keys)
    print(f'load_state_dict: {n_missing} missing keys, {n_unexpected} unexpected keys')
    if n_missing:
        print(f'  missing (first 10): {load_result.missing_keys[:10]}')
    if n_unexpected:
        print(f'  unexpected (first 10): {load_result.unexpected_keys[:10]}')
    max_allowed_mismatch = 0 if args.base == 'composed' else 5
    assert n_missing + n_unexpected <= max_allowed_mismatch, (
        f'{n_missing} missing + {n_unexpected} unexpected keys when loading '
        f'{args.checkpoint} onto {base_desc} -- structural mismatch, would silently '
        f"report the base model's performance, not this checkpoint's."
    )
    model.cuda().eval()

    samples = [json.loads(l) for l in open(args.test_manifest)]
    if args.max_samples > 0: samples = samples[:args.max_samples]
    print(f'Generating {args.num_beams}-best for {len(samples)} samples...')

    # Passing do_sample=False (or any field matching the generation
    # library's own class default) nested inside a GenerationConfig
    # object does not reliably stick for the --base composed path: the
    # underlying library backfills any field left equal to the class
    # default from the language model's own generation_config.json, which
    # for this LLM sets do_sample=True with non-trivial temperature/top_k/
    # top_p by default, silently turning intended beam search into
    # sampling. Passing these as direct keyword arguments to generate()
    # applies after that backfill and reliably wins -- verified via a
    # repeat-run determinism check (identical output across repeated runs
    # of the same samples).
    gen_kwargs = dict(
        bos_token_id=model.text_bos_id,
        eos_token_id=model.text_eos_id,
        pad_token_id=model.text_pad_id,
        num_beams=args.num_beams,
        num_return_sequences=args.num_beams,
        do_sample=False,
        temperature=None, top_k=None, top_p=None,
        early_stopping=True,
        output_scores=True,
        return_dict_in_generate=True,
    )

    results, errors = [], 0
    with torch.no_grad():
        for i, s in enumerate(samples):
            if (i + 1) % 200 == 0:
                print(f'  {i+1}/{len(samples)} ({errors} errors so far)')
            try:
                out = model.generate(
                    prompts=[[{'role': 'user',
                        'content': f'Transcribe the following: {model.audio_locator_tag}',
                        'audio': [s['audio_filepath']]}]],
                    max_new_tokens=128, **gen_kwargs)
                seqs = out.sequences.cpu()
                scores = out.sequences_scores.cpu().tolist()
                hyps = [model.tokenizer.ids_to_text(seq).lower().strip() for seq in seqs]
                results.append({
                    'audio_filepath': s['audio_filepath'],
                    'text': s['text'].lower().strip(),
                    'hyps': [{'text': h, 'model_score': sc} for h, sc in zip(hyps, scores)],
                })
            except Exception as e:
                errors += 1
                print(f'  [error @ {i}] {e}')

    print(f'\nDone: {len(results)} samples, {errors} errors')
    json.dump({'num_beams': args.num_beams, 'errors': errors, 'results': results},
               open(args.output, 'w'), indent=2)
    print(f'Saved to {args.output}')

if __name__ == '__main__': main()
