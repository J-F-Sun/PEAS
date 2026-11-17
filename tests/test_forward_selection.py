import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from peas.common.paths import repository_root
from peas.ablations.selection import select_forward_epochs

class ForwardSelectionTests(TestCase):
    def test_recorded_validation_reconstructs_selected_epochs(self):
        root=repository_root()
        records=json.loads((root/'data/validation/canonical_forward_scores.json').read_text())
        expected=json.loads((root/'configs/ablations/canonical_forward_selection.json').read_text())['epochs']
        with TemporaryDirectory() as directory:
            for record in records:
                p=Path(directory)/'validation/no_reformulation/canonical'/f"seed_{record['seed']}"/record['source_id']
                p.mkdir(parents=True)
                (p/'validation.json').write_text(json.dumps({'scores':record['scores']}))
                (p/'status.json').write_text(json.dumps({'phase':'complete'}))
                (p/'summary.json').write_text(json.dumps({'smoke':False}))
            self.assertEqual(select_forward_epochs(directory,list(map(int,expected)))['epochs'],expected)
            (p/'status.json').write_text(json.dumps({'phase':'adam'}))
            with self.assertRaises(ValueError):select_forward_epochs(directory,list(map(int,expected)))
