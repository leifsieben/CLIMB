import tempfile
import unittest
from pathlib import Path

from experiment_manifest import (
    DEFAULT_SUPERVISED_FAMILIES,
    _balanced_representative_sequences,
    generate_manifest,
)


class ExperimentManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.results_root = Path(self.tmpdir.name) / "results"
        self.spec = {
            "name": "test",
            "results_root": str(self.results_root),
            "tokenizer_path": "./tokenizer",
            "unsupervised_data": ["./raw_data/pubchem_filtered"],
            "supervised_parquet_path": "./supervised_wide.parquet",
            "supervised_tokenized_parquet_path": "./supervised_wide.parquet",
            "supervised_families": DEFAULT_SUPERVISED_FAMILIES,
            "model": {
                "hidden_size": 256,
                "num_hidden_layers": 6,
                "num_attention_heads": 8,
                "intermediate_size": 1024,
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "max_position_embeddings": 512,
            },
            "compute_budget_defaults": {
                "total_epochs": 1,
                "supervised_fraction": 0.5,
                "total_tokens": 10_000_000_000,
            },
            "mlm_training": {
                "batch_size": 32,
                "learning_rate": 5e-5,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "logging_steps": 10,
                "save_steps": 20,
                "eval_steps": 0,
                "fp16": False,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0,
                "save_total_limit": 2,
                "dataloader_num_workers": 0,
                "evaluation_strategy": "no",
                "shuffle": True,
            },
            "supervised_training": {
                "output_dir": str(self.results_root),
                "num_epochs": 1,
                "batch_size": 32,
                "learning_rate": 2e-5,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "eval_steps": 0,
                "save_steps": 20,
                "logging_steps": 10,
                "fp16": False,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0,
                "dataloader_num_workers": 0,
                "streaming_batch_rows": 1024,
            },
            "validation_fraction": 0.0,
            "evaluation": {},
            "cluster": {},
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_balanced_sequences_are_unique(self):
        seqs = _balanced_representative_sequences(
            ["PCQM", "WONG", "L1000_MCF7", "L1000_VCAP", "PCBA"],
            k=3,
            n_sequences=5,
            seed=17,
        )
        self.assertEqual(len(seqs), 5)
        self.assertEqual(len(set(seqs)), 5)

    def test_manifest_counts_match_plan(self):
        manifest = generate_manifest(self.spec)
        self.assertEqual(manifest["summary"]["expected"]["unsupervised_baseline"], 15)
        self.assertEqual(manifest["summary"]["expected"]["supervised_order_ramp"], 25)
        self.assertEqual(manifest["summary"]["expected"]["unsupervised_fixed_budget"], 5)
        self.assertEqual(manifest["summary"]["expected"]["mixed_fixed_budget"], 15)
        self.assertEqual(manifest["summary"]["smoke_runs"], 3)
        self.assertEqual(len(manifest["runs"]), 63)

    def test_manifest_runs_embed_repro_metadata(self):
        manifest = generate_manifest(self.spec)
        sample = manifest["runs"][0]
        self.assertIn("pretrain_config", sample)
        self.assertIn("run_metadata", sample["pretrain_config"])
        self.assertEqual(sample["pretrain_config"]["run_metadata"]["run_type"], sample["run_type"])


if __name__ == "__main__":
    unittest.main()
