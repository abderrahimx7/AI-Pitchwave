import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import build_match, parse_events, feature_vector, FEATURE_NAMES, split_matches


def event(second, kind='Pass', team=1, period=1, location=None):
    minutes, secs = divmod(second, 60)
    e = {'period': period, 'timestamp': f'00:{minutes:02d}:{secs:06.3f}', 'type': {'name': kind}, 'team': {'id': team}}
    if location is not None:
        e['location'] = location
    return e


def sample():
    return [event(20, location=[60,40]), event(30, 'Shot'), event(40, 'Shot'),
            event(41, 'Shot', 2), event(80, 'Half End'),
            event(5, 'Shot', period=2), event(80, 'Half End', period=2)]


class FeatureTests(unittest.TestCase):
    def test_label_boundaries(self):
        X, y, meta = build_match(sample(), 1, 2)
        at30 = (meta[:,0] == 1) & (meta[:,1] == 30)
        self.assertEqual(y[at30].tolist(), [1,0])  # 40 included, 41 excluded.
        raw = [e for e in sample() if not (e['period']==1 and e['timestamp']=='00:00:40.000')]
        _, y2, meta2 = build_match(raw, 1, 2)
        self.assertEqual(y2[(meta2[:,0]==1)&(meta2[:,1]==30)].tolist(), [0,0])  # Shot at t is not future.

    def test_future_changes_do_not_change_features(self):
        raw = sample()
        before = feature_vector([e for e in parse_events(raw) if e['period']==1],1,2,30,1)
        raw += [event(31,'Shot',1,location=[119,40]), event(45,'Carry',2,location=[90,10])]
        after = feature_vector([e for e in parse_events(raw) if e['period']==1],1,2,30,1)
        np.testing.assert_array_equal(before, after)

    def test_outcome_metadata_is_ignored(self):
        raw = sample(); changed = copy.deepcopy(raw)
        changed[0].update({'duration':9999, 'possession':777, 'shot':{'statsbomb_xg':.99}, 'pass':{'end_location':[120,40]}})
        np.testing.assert_array_equal(build_match(raw,1,2)[0], build_match(changed,1,2)[0])

    def test_periods_do_not_mix(self):
        X, y, meta = build_match(sample(),1,2)
        row = X[(meta[:,0]==2)&(meta[:,1]==30)&(meta[:,2]==1)][0]
        self.assertEqual(row[FEATURE_NAMES.index('own_count_shot')],1)
        self.assertEqual(row[FEATURE_NAMES.index('own_count_pass')],0)
        self.assertLessEqual(meta[:,1].max(),70)

    def test_missing_end_rejected(self):
        with self.assertRaises(ValueError):
            build_match([event(20)],1,2)

    def test_extra_time_is_excluded(self):
        a = build_match(sample(),1,2)
        b = build_match(sample()+[event(40,'Shot',period=3)],1,2)
        np.testing.assert_array_equal(a[0],b[0]); np.testing.assert_array_equal(a[1],b[1])

    def test_missing_location_and_no_history(self):
        v = feature_vector([],1,2,30,1)
        self.assertEqual(len(v),28)
        self.assertTrue(np.isfinite(v).all())
        self.assertEqual(v[FEATURE_NAMES.index('own_has_location')],0)

    def test_split_is_disjoint_and_ordered(self):
        split = split_matches(list(range(64)))
        self.assertEqual([len(split[k]) for k in ('train','val','test')],[44,10,10])
        self.assertFalse(set(split['train']) & set(split['test']))
        self.assertLess(max(split['train']), min(split['val']))
        self.assertLess(max(split['val']), min(split['test']))

    def test_too_few_matches(self):
        with self.assertRaises(ValueError): split_matches(list(range(5)))

    def test_positive_duration(self):
        with self.assertRaises(ValueError): build_match(sample(),1,2,step=0)


if __name__ == '__main__':
    unittest.main()
