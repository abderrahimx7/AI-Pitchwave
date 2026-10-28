"""Export a held-out match timeline; does NOT manufacture a spatial pressure map."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from features import FEATURE_NAMES
from model import ShotNet, choose_device
from train import predict

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, default=ROOT/'prepared')
    p.add_argument('--run', type=Path, default=ROOT/'runs'/'run1')
    p.add_argument('--output', type=Path, default=ROOT/'predictions.json')
    p.add_argument('--match-id', type=int)
    p.add_argument('--model', choices=['nn', 'logistic'], default='nn')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a = p.parse_args()
    config = json.loads((a.prepared/'config.json').read_text(encoding='utf-8'))
    data = dict(np.load(a.prepared/'test.npz', allow_pickle=False))
    mid = a.match_id if a.match_id is not None else int(data['match_id'][0])
    mask = data['match_id'] == mid
    if not mask.any():
        raise SystemExit('Select a TEST match ID from prepared/config.json. Training-match predictions are intentionally excluded.')
    X = data['X'][mask]
    if a.model == 'nn':
        # Load only your own checkpoint; weights_only rejects unrestricted pickle content.
        ckpt = torch.load(a.run/'model.pt', map_location='cpu', weights_only=True)
        if ckpt['feature_names'] != FEATURE_NAMES or any(ckpt[k] != config[k] for k in ('window', 'horizon', 'step')):
            raise ValueError('Model/data configuration mismatch')
        device = choose_device(a.device)
        model = ShotNet(ckpt['n_features']).to(device)
        model.load_state_dict(ckpt['state_dict'])
        X = np.ascontiguousarray((X-ckpt['mean'].numpy())/ckpt['std'].numpy(), dtype=np.float32)
        prob = predict(model, X, device)
    else:
        ckpt = dict(np.load(a.run/'logistic.npz', allow_pickle=False))
        report = json.loads((a.run/'metrics.json').read_text(encoding='utf-8'))
        if any(report['config'][k] != config[k] for k in ('feature_names', 'window', 'horizon', 'step')):
            raise ValueError('Model/data configuration mismatch')
        z = ((X-ckpt['mean'])/ckpt['std']) @ ckpt['coef'].T + ckpt['intercept']
        prob = (1/(1+np.exp(-np.clip(z, -50, 50)))).ravel()
    match = next(m for m in config['source_manifest']['matches'] if m['match_id'] == mid)
    frames = {}
    for row, pr in zip(data['meta'][mask], prob):
        period, sec, team = int(row[0]), float(row[1]), int(row[2])
        key = (period, sec)
        frame = frames.setdefault(key, {'period': period, 'period_seconds': sec})
        side = 'home' if team == match['home_team']['home_team_id'] else 'away'
        frame[f'{side}_shot_probability'] = float(pr)
    result = {'schema': 'pitchwave-shot-risk-v1', 'source': 'StatsBomb Open Data',
              'model': a.model, 'match_id': mid, 'home': match['home_team']['home_team_name'],
              'away': match['away_team']['away_team_name'], 'horizon_seconds': config['horizon'],
              'notice': 'Independent estimated shot risks, not complementary shares; no spatial prediction. No extra time. Period clock includes stoppage time.',
              'frames': list(frames.values())}
    if any(not ('home_shot_probability' in f and 'away_shot_probability' in f) for f in result['frames']):
        raise ValueError('Incomplete team pair')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('Saved', a.output, 'frames:', len(result['frames']), 'match:', mid)


if __name__ == '__main__':
    main()
