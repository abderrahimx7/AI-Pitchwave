"""Download selected official StatsBomb event files; no API key required."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BASE = 'https://raw.githubusercontent.com/hudl/open-data/master/'


def fetch(relative, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return json.loads(destination.read_text(encoding='utf-8'))
    for attempt in range(3):
        try:
            with urlopen(Request(BASE + relative, headers={'User-Agent': 'Pitchwave-Learning-Project/1.0'}), timeout=60) as response:
                raw = response.read()
            value = json.loads(raw)
            temporary = destination.with_suffix('.part')
            temporary.write_bytes(raw)
            temporary.replace(destination)
            return value
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--competition', type=int, default=43)
    parser.add_argument('--season', type=int, default=106)
    parser.add_argument('--max-matches', type=int, default=64)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    if args.max_matches < 1:
        parser.error('--max-matches must be positive')
    folder = args.data_dir
    competitions = fetch('data/competitions.json', folder / 'competitions.json')
    selected = [c for c in competitions if c['competition_id'] == args.competition and c['season_id'] == args.season]
    if not selected:
        raise SystemExit('Competition/season not in current catalogue. See data/competitions.json.')
    print(selected[0]['competition_name'], selected[0]['season_name'], flush=True)
    matches = fetch(f'data/matches/{args.competition}/{args.season}.json', folder / f'matches_{args.competition}_{args.season}.json')
    matches = sorted(matches, key=lambda m: (m['match_date'], m.get('kick_off') or '', m['match_id']))[:args.max_matches]
    entries = []
    for i, match in enumerate(matches, 1):
        mid = match['match_id']
        file = folder / 'events' / f'{mid}.json'
        events = fetch(f'data/events/{mid}.json', file)
        if not isinstance(events, list) or not events:
            raise ValueError(f'Unexpected events payload for {mid}')
        entries.append({**match, 'event_file': f'events/{mid}.json', 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
        print(f'[{i}/{len(matches)}] {mid}: {len(events)} events', flush=True)
    manifest = {'source': 'StatsBomb Open Data', 'repository': 'https://github.com/hudl/open-data',
                'competition': selected[0], 'matches': entries}
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved manifest:', folder / 'manifest.json')


if __name__ == '__main__':
    main()
