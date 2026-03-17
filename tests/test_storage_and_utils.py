import json
import tempfile
import unittest
from pathlib import Path

from storage_utils import is_s3_uri, list_data_files, split_s3_uri
from token_budget import TokenBudgetTracker
from utils import find_latest_checkpoint


class StorageAndUtilsTests(unittest.TestCase):
    def test_split_s3_uri(self):
        bucket, key = split_s3_uri("s3://my-bucket/path/to/file.parquet")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(key, "path/to/file.parquet")
        self.assertTrue(is_s3_uri("s3://my-bucket/x"))

    def test_list_local_data_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.pkl").write_text("x")
            (root / "b.parquet").write_text("x")
            resolved = list_data_files(str(root))
            self.assertEqual(len(resolved), 2)

    def test_token_tracker_counts_attention_mask(self):
        tracker = TokenBudgetTracker(token_budget=None)
        tracker.update({"attention_mask": [[1, 1, 0], [1, 0, 0]]})
        self.assertEqual(tracker.tokens_seen, 3)

    def test_find_latest_checkpoint_prefers_highest_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "checkpoint-10").mkdir()
            (root / "checkpoint-20").mkdir()
            latest = find_latest_checkpoint(str(root))
            self.assertTrue(latest.endswith("checkpoint-20"))

    def test_find_latest_checkpoint_prefers_spot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spot = root / "spot_checkpoint"
            spot.mkdir()
            (spot / "trainer_state.json").write_text(json.dumps({}))
            (root / "checkpoint-30").mkdir()
            latest = find_latest_checkpoint(str(root))
            self.assertTrue(latest.endswith("spot_checkpoint"))


if __name__ == "__main__":
    unittest.main()
