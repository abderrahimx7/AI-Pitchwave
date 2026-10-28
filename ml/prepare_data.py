"""Process one match at a time; persist reproducible match splits and feature metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from features import build_match, FEATURE_NAMES, split_matches

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT/'data')
    p.add_argument('--output', type=Path, default=ROOT/'prepared')
    p.add_argument('--window', type=int, default=30)
    p.add_argument('--horizon', type=int, default=10)
    p.add_argument('--step', type=int, default=5)
    a = p.parse_args()
    if min(a.window, a.horizon, a.step) < 1:
        p.error('Durations must be positive')
    manifest = json.loads((a.data_dir/'manifest.json').read_text(encoding='utf-8'))
    matches = sorted(manifest['matches'], key=lambda m: (m['match_date'], m.get('kick_off') or '', m['match_id']))
    splits = split_matches([m['match_id'] for m in matches])
    groups = {name: [] for name in splits}
    for m in matches:
        mid = m['match_id']
        file = a.data_dir/m['event_file']
        if hashlib.sha256(file.read_bytes()).hexdigest() != m['sha256']:
            raise ValueError(f'Event file changed: {mid}. Download/manifest must agree.')
        raw = json.loads(file.read_text(encoding='utf-8'))
        X, y, meta = build_match(raw, m['home_team']['home_team_id'], m['away_team']['away_team_id'], a.window, a.horizon, a.step)
        if not len(y):
            raise ValueError(f'No usable windows: {mid}')
        group = next(name for name, ids in splits.items() if mid in ids)
        groups[group].append((X, y, meta, np.full(len(y), mid, dtype=np.int64)))
        print(mid, group, len(y), 'rows, positives:', int(y.sum()), flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, chunks in groups.items():
        arrays = [np.concatenate([part[k] for part in chunks]) for k in range(4)]
        X, y, meta, mids = arrays
        if not np.isfinite(X).all() or len(np.unique(y)) != 2:
            raise ValueError(f'{name} needs finite features and both label classes; add matches')
        np.savez_compressed(a.output/f'{name}.npz', X=X, y=y, meta=meta, match_id=mids)
        stats[name] = {'rows': len(y), 'positives': int(y.sum()), 'positive_rate': float(y.mean()), 'matches': len(splits[name])}
    config = {'feature_names': FEATURE_NAMES, 'window': a.window, 'horizon': a.horizon, 'step': a.step,
              'target': 'team shot starts in (t,t+horizon], same half', 'splits': splits,
              'stats': stats, 'source_manifest': manifest}
    (a.output/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
