"""Causal features and future-shot labels. No torch dependency; easily testable."""
import bisect
import math
import numpy as np

TYPES = ('Pass', 'Carry', 'Shot', 'Pressure', 'Dribble', 'Ball Recovery')
LOCATION_TYPES = {'Pass', 'Carry', 'Shot', 'Dribble'}
TEAM_FEATURES = [f'count_{name.lower().replace(" ", "_")}' for name in TYPES] + [
    'selected_events', 'age_last_event', 'mean_x', 'mean_y', 'last_x', 'last_y', 'has_location']
FEATURE_NAMES = [f'{prefix}_{name}' for prefix in ('own', 'opp') for name in TEAM_FEATURES] + ['period', 'period_seconds']


def seconds(timestamp):
    h, m, s = timestamp.split(':')
    value = int(h) * 3600 + int(m) * 60 + float(s)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Invalid event timestamp')
    return value


def parse_events(raw):
    """Discard fields that can encode future outcomes (xG, possession IDs, duration...)."""
    parsed = []
    for e in raw:
        if e.get('period') not in (1, 2):
            continue  # Exclude extra time and penalty shootouts in version 1.
        if not e.get('timestamp'):
            raise ValueError('Missing timestamp in first/second-half event')
        location = e.get('location')
        if location is not None:
            if len(location) < 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in location[:2]):
                raise ValueError('Invalid location')
            location = [float(location[0]), float(location[1])]
            if not (0 <= location[0] <= 120 and 0 <= location[1] <= 80):
                location = None  # Ignore out-of-pitch records, not silently clip them.
        parsed.append({'period': e['period'], 't': seconds(e['timestamp']),
                       'type': e.get('type', {}).get('name', ''),
                       'team': e.get('team', {}).get('id'), 'location': location})
    return sorted(parsed, key=lambda e: (e['period'], e['t']))


def team_summary(history, team, t, window):
    records = [e for e in history if e['team'] == team and e['type'] in TYPES]
    counts = [sum(e['type'] == kind for e in records) for kind in TYPES]
    age = (t - records[-1]['t']) / window if records else 1.0
    located = [e for e in records if e['type'] in LOCATION_TYPES and e['location'] is not None]
    if located:
        points = np.asarray([e['location'] for e in located], dtype=np.float32) / [120., 80.]
        mean = points.mean(axis=0).tolist()
        last = points[-1].tolist()
    else:
        mean, last = [0., 0.], [0., 0.]
    return counts + [len(records), age] + mean + last + [float(bool(located))]


def feature_vector(period_events, team, opponent, t, period, window=30):
    # Event starts in [t-window,t] are observable. No event-end attributes used.
    history = [e for e in period_events if t - window <= e['t'] <= t]
    return team_summary(history, team, t, window) + team_summary(history, opponent, t, window) + [period, t / 2700.]


def build_match(raw, home_id, away_id, window=30, horizon=10, step=5):
    if min(window, horizon, step) <= 0:
        raise ValueError('window, horizon and step must be positive')
    events = parse_events(raw)
    X, y, meta = [], [], []
    for period in (1, 2):
        pe = [e for e in events if e['period'] == period]
        ends = [e['t'] for e in pe if e['type'] == 'Half End']
        if not ends:
            raise ValueError(f'Missing Half End for period {period}; cannot label censored windows safely')
        end = min(ends)
        times = [e['t'] for e in pe]
        shot_times = {team: [e['t'] for e in pe if e['team'] == team and e['type'] == 'Shot'] for team in (home_id, away_id)}
        # Skip warm-up and the final horizon seconds of EACH HALF.
        for t in np.arange(window, end - horizon + 1e-7, step):
            lo, hi = bisect.bisect_left(times, t-window), bisect.bisect_right(times, t)
            history = pe[lo:hi]
            for team, opp in ((home_id, away_id), (away_id, home_id)):
                X.append(feature_vector(history, team, opp, float(t), period, window))
                shots = shot_times[team]
                start = bisect.bisect_right(shots, t)
                finish = bisect.bisect_right(shots, t+horizon)
                y.append(int(finish > start))  # Strictly future: (t,t+horizon].
                meta.append((period, float(t), team))
    return np.asarray(X, dtype=np.float32).reshape(-1, len(FEATURE_NAMES)), np.asarray(y, dtype=np.float32), np.asarray(meta, dtype=np.float64).reshape(-1, 3)


def split_matches(ordered_ids):
    """Chronological match-level split, with at least two matches per held-out set."""
    ids = list(ordered_ids)
    if len(ids) < 12 or len(set(ids)) != len(ids):
        raise ValueError('Need at least 12 distinct matches; use 64 for the initial experiment')
    n_test = max(2, round(len(ids)*.15))
    n_val = max(2, round(len(ids)*.15))
    return {'train': ids[:-(n_test+n_val)], 'val': ids[-(n_test+n_val):-n_test], 'test': ids[-n_test:]}
