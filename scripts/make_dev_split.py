#!/usr/bin/env python3
"""Carve a held-out development split out of the UWB-ATCC training set,
grouped by recording session (not by individual cut), so no session's
utterances are split across train and dev.

Why this exists: an earlier configuration's validation set pointed at the
same file used for final test-set WER reporting, meaning checkpoint
selection was effectively being made on what should be held-out test
data. This script produces a properly separated dev split from the
training data instead, and does not touch the test set.

Grouping unit: UWB-ATCC cut IDs follow the pattern
    uwb-atcc_<POSITION>-<SESSIONHASH>_<start_ms>_<end_ms>_<AT|PI>
e.g. uwb-atcc_APP-lnNSkN_000056_000423_AT. The `<POSITION>-<SESSIONHASH>`
prefix identifies one continuous recorded session; grouping is done on
this prefix (not on the corpus's per-cut recording_id field, which is
unique per cut and would not prevent same-session leakage across the
split).

This script also excludes one session found to be present in both the
original training and test manifests (verified directly by session-prefix
intersection) -- an independent train/test leak in the source data,
resolved here as the split is rebuilt.

Usage:
    python make_dev_split.py \
        --train-cuts <data-dir>/train_cuts.jsonl.gz \
        --test-cuts <data-dir>/test_cuts.jsonl.gz \
        --out-dir <data-dir> \
        --dev-fraction 0.08 --seed 1234
"""
import argparse
import random
import re
from pathlib import Path

import lhotse

SESSION_PAT = re.compile(r"^(uwb-atcc_[A-Za-z]+-[A-Za-z0-9]+)_\d+_\d+_[A-Z]+$")


def session_of(cut_id: str) -> str:
    m = SESSION_PAT.match(cut_id)
    if not m:
        raise ValueError(f"cut id does not match expected UWB-ATCC pattern: {cut_id!r}")
    return m.group(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cuts", required=True)
    p.add_argument("--test-cuts", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dev-fraction", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    train_cuts = lhotse.CutSet.from_file(args.train_cuts)
    test_cuts = lhotse.CutSet.from_file(args.test_cuts)

    test_sessions = {session_of(c.id) for c in test_cuts}

    by_session = {}
    for c in train_cuts:
        by_session.setdefault(session_of(c.id), []).append(c)

    session_hours_all = {s: sum(c.duration for c in cuts) / 3600 for s, cuts in by_session.items()}

    leaked = sorted(set(by_session) & test_sessions)
    if leaked:
        print(f"Excluding {len(leaked)} train session(s) that also appear in test_cuts.jsonl.gz: {leaked}")
        for s in leaked:
            del by_session[s]

    sessions = sorted(by_session)  # sort first for determinism, then shuffle with a fixed seed
    rng = random.Random(args.seed)
    rng.shuffle(sessions)

    session_hours = {s: session_hours_all[s] for s in by_session}
    total_hours = sum(session_hours.values())
    target_dev_hours = total_hours * args.dev_fraction

    dev_sessions, dev_hours = [], 0.0
    for s in sessions:
        if dev_hours >= target_dev_hours:
            break
        dev_sessions.append(s)
        dev_hours += session_hours[s]
    dev_sessions = set(dev_sessions)
    train_sessions = set(sessions) - dev_sessions

    dev_cuts = [c for s in dev_sessions for c in by_session[s]]
    new_train_cuts = [c for s in train_sessions for c in by_session[s]]

    # dev_sessions and train_sessions are constructed disjoint from the
    # same pool above; the real protection against a session appearing in
    # both a split and the test set is the exclusion loop above. These
    # pairwise assertions are the guarantee kept for future reruns of this
    # script against different source data.
    assert not (dev_sessions & train_sessions)
    assert not (dev_sessions & test_sessions)
    assert not (train_sessions & test_sessions)

    out_dir = Path(args.out_dir)
    dev_path = out_dir / "dev_cuts.jsonl.gz"
    train_path = out_dir / "train_cuts_v2.jsonl.gz"
    lhotse.CutSet.from_cuts(dev_cuts).to_file(dev_path)
    lhotse.CutSet.from_cuts(new_train_cuts).to_file(train_path)

    print(f"Original train: {len(train_cuts)} cuts, {sum(session_hours_all.values()):.3f}h "
          f"({len(by_session) + len(leaked)} sessions incl. {len(leaked)} excluded-leaked)")
    print(f"New train:      {len(new_train_cuts)} cuts, {sum(c.duration for c in new_train_cuts)/3600:.3f}h, "
          f"{len(train_sessions)} sessions -> {train_path}")
    print(f"Dev:            {len(dev_cuts)} cuts, {sum(c.duration for c in dev_cuts)/3600:.3f}h, "
          f"{len(dev_sessions)} sessions -> {dev_path}")
    print(f"Test (untouched): {len(test_cuts)} cuts, {sum(c.duration for c in test_cuts)/3600:.3f}h, "
          f"{len(test_sessions)} sessions -> {args.test_cuts}")


if __name__ == "__main__":
    main()
