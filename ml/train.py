"""Train a neural network and compare it with constant and logistic baselines."""
import argparse
import copy
import json
from pathlib import Path
import platform
import random
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model import ShotNet, choose_device

ROOT = Path(__file__).resolve().parent


def metrics(y, probability):
    p = np.clip(probability.astype(float), 1e-7, 1-1e-7)
    bins = []
    bin_ids = np.minimum((p * 10).astype(int), 9)
    for index in range(10):
        lo = index / 10
        mask = bin_ids == index
        if mask.any():
            bins.append({'lower': round(float(lo), 2), 'n': int(mask.sum()),
                         'mean_prediction': float(p[mask].mean()), 'observed_rate': float(y[mask].mean())})
    return {'average_precision': float(average_precision_score(y, p)),
            'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'positive_rate': float(y.mean()), 'mean_prediction': float(p.mean()), 'calibration_bins': bins}


def predict(model, X, device, batch_size=1024):
    model.eval()
    result = []
    with torch.no_grad():
        for offset in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[offset:offset+batch_size]).to(device)
            result.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, default=ROOT/'prepared')
    p.add_argument('--output', type=Path, default=ROOT/'runs'/'run1')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1:
        p.error('epochs and batch-size must be positive')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Output folder already contains results. Choose --output runs/run2 to preserve the previous experiment.')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(2)
    device = choose_device(a.device)
    config = json.loads((a.prepared/'config.json').read_text(encoding='utf-8'))
    data = {s: dict(np.load(a.prepared/f'{s}.npz', allow_pickle=False)) for s in ('train', 'val', 'test')}
    group_sets = [set(data[s]['match_id'].tolist()) for s in ('train', 'val', 'test')]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Match overlap between splits')
    # Fit normalisation using TRAIN ONLY.
    mean = data['train']['X'].mean(axis=0)
    std = data['train']['X'].std(axis=0)
    std[std < 1e-6] = 1.0
    X = {s: np.ascontiguousarray((data[s]['X']-mean)/std, dtype=np.float32) for s in data}
    y = {s: data[s]['y'] for s in data}
    for s in data:
        if len(np.unique(y[s])) != 2 or not np.isfinite(X[s]).all():
            raise ValueError(f'Invalid {s} arrays: need finite features and both classes')
    logistic = LogisticRegression(max_iter=2000, random_state=a.seed)
    logistic.fit(X['train'], y['train'])
    model = ShotNet(X['train'].shape[1]).to(device)
    # Start near the training prior, not a misleading 50% rare-event prediction.
    prior = float(y['train'].mean())
    with torch.no_grad():
        model.layers[-1].bias.fill_(np.log(prior/(1-prior)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    loss_fn = nn.BCEWithLogitsLoss()  # No resampling/positive weighting: keep natural class prevalence.
    generator = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(X['train']), torch.from_numpy(y['train'])),
                        batch_size=a.batch_size, shuffle=True, num_workers=0, generator=generator,
                        pin_memory=device.type == 'cuda')
    print('Device:', device, torch.cuda.get_device_name(0) if device.type == 'cuda' else '')
    print('Parameters:', sum(p.numel() for p in model.parameters()), 'training prior:', prior)
    best_loss, best, stale, history = float('inf'), None, 0, []
    for epoch in range(1, a.epochs+1):
        model.train(); total_loss = 0.
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(yb)
        val_p = predict(model, X['val'], device)
        val = metrics(y['val'], val_p)
        history.append({'epoch': epoch, 'train_bce': total_loss/len(y['train']), 'val_log_loss': val['log_loss'], 'val_average_precision': val['average_precision']})
        print(f"Epoch {epoch:02d}: train={history[-1]['train_bce']:.5f} val={val['log_loss']:.5f} AP={val['average_precision']:.5f}", flush=True)
        if val['log_loss'] < best_loss - 1e-5:
            best_loss, best, stale = val['log_loss'], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(best)
    # Test is used only after epoch selection is complete.
    report = {'target': config['target'], 'config': config, 'device': str(device), 'seed': a.seed,
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'torch': str(torch.__version__)},
              'history': history, 'evaluation': {}}
    for s in ('val', 'test'):
        network_p = predict(model, X[s], device)
        report['evaluation'][s] = {
            'constant': metrics(y[s], np.full(len(y[s]), prior)),
            'logistic': metrics(y[s], logistic.predict_proba(X[s])[:, 1]),
            'neural_network': metrics(y[s], network_p)}
    a.output.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({'state_dict': state, 'mean': torch.from_numpy(mean), 'std': torch.from_numpy(std),
                'n_features': len(mean), 'feature_names': config['feature_names'],
                'window': config['window'], 'horizon': config['horizon'], 'step': config['step']}, a.output/'model.pt')
    np.savez_compressed(a.output/'logistic.npz', coef=logistic.coef_, intercept=logistic.intercept_, mean=mean, std=std)
    (a.output/'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('\nTEST (single held-out evaluation):')
    for name, m in report['evaluation']['test'].items():
        print(name, {k: round(m[k], 5) for k in ('average_precision', 'brier', 'log_loss')})
    print('Saved:', a.output, '\nProbabilities are estimates, not calibrated guarantees; inspect metrics/calibration before claims.')


if __name__ == '__main__':
    main()
