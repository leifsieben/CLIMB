#!/usr/bin/env python
"""
run_pipeline.py
===============
Master script to run the complete training and evaluation pipeline

Steps:
1. Hyperparameter search on unsupervised data
2. Train model with best hyperparameters
3. Evaluate on MoleculeNet benchmarks

Usage:
    python run_pipeline.py --config config.yaml
"""

import argparse
import subprocess
import json
import logging
from pathlib import Path
import sys
from utils import setup_logging

logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    logger.info(f"\n{'='*80}")
    logger.info(f"{description}")
    logger.info(f"{'='*80}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        logger.error(f"❌ Failed: {description}")
        sys.exit(1)
    
    logger.info(f"✓ Completed: {description}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run complete training pipeline")
    parser.add_argument("--config", default="./experiments/config.yaml", help="Config file")
    parser.add_argument("--tokenizer", default="local_prototyping_data/tokenizer", help="Tokenizer path")
    parser.add_argument("--unsup_data", default="local_prototyping_data/unsupervised_tokenized.pkl", help="Unsupervised data")
    parser.add_argument("--sup_data", default="local_prototyping_data/supervised_tokenized.pkl", help="Supervised data")
    parser.add_argument("--hp_trials", type=int, default=20, help="Number of HP search trials")
    parser.add_argument("--skip_hp_search", action="store_true", help="Skip HP search, use existing results")
    parser.add_argument("--output_dir", default="experiments", help="Output directory")
    args = parser.parse_args()
    
    setup_logging()
    
    logger.info("="*80)
    logger.info("CHEMICAL LANGUAGE MODEL - COMPLETE PIPELINE")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Tokenizer: {args.tokenizer}")
    logger.info(f"Unsupervised data: {args.unsup_data}")
    logger.info(f"Supervised data: {args.sup_data}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths
    hp_output = output_dir / "hp_search"
    model_output = output_dir / "model_sup_50_unsup_50"
    eval_output = output_dir / "evaluation"
    
    # =========================================================================
    # STEP 1: Hyperparameter Search
    # =========================================================================
    
    if not args.skip_hp_search:
        hp_cmd = [
            "python", "hyperparameter_search.py",
            "--config", args.config,
            "--tokenizer", args.tokenizer,
            "--train_data", args.unsup_data,
            "--output", str(hp_output),
            "--n_trials", str(args.hp_trials),
        ]
        
        run_command(hp_cmd, "STEP 1: Hyperparameter Search")
        
        # Load best hyperparameters
        best_params_file = hp_output / "best_hyperparameters.json"
        with open(best_params_file) as f:
            best_params = json.load(f)
        
        logger.info(f"\n✓ Best hyperparameters found:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        # Update config with best hyperparameters
        logger.info(f"\n⚠️  IMPORTANT: Update your config.yaml with these hyperparameters!")
        logger.info(f"  Or re-run with the updated config for training.")
    else:
        logger.info("Skipping hyperparameter search (using existing config)")
    
    # =========================================================================
    # STEP 2: Train Model (50% Supervised, 50% Unsupervised)
    # =========================================================================
    
    train_cmd = [
        "python", "train_model.py",
        "--config", args.config,
        "--tokenizer", args.tokenizer,
        "--unsup_data", args.unsup_data,
        "--unsup_weight", "0.50",
        "--sup_weight", "0.50",
        "--output", str(model_output),
        "--task", "mlm",
    ]
    
    run_command(train_cmd, "STEP 2: Train Model (100% Unsupervised)")
    
    # =========================================================================
    # STEP 3: Evaluate on MoleculeNet
    # =========================================================================
    
    # Evaluate on multiple MoleculeNet datasets
    moleculenet_datasets = ["BBBP", "ESOL", "FreeSolv", "Lipophilicity"]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP 3: Evaluate on MoleculeNet Benchmarks")
    logger.info(f"{'='*80}")
    logger.info(f"Datasets: {', '.join(moleculenet_datasets)}")
    
    eval_results = {}
    
    for dataset_name in moleculenet_datasets:
        dataset_output = eval_output / dataset_name.lower()
        
        eval_cmd = [
            "python", "evaluate_model.py",
            "--pretrained_model", str(model_output),
            "--dataset", "moleculenet",
            "--dataset_name", dataset_name,
            "--output", str(dataset_output),
            "--num_epochs", "50",
            "--batch_size", "32",
        ]
        
        try:
            run_command(eval_cmd, f"Evaluating on {dataset_name}")
            
            # Load results
            results_file = dataset_output / "results.json"
            with open(results_file) as f:
                results = json.load(f)
            
            eval_results[dataset_name] = results['metrics']
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to evaluate on {dataset_name}: {e}")
            logger.warning(f"   (Dataset may not be available or DeepChem issue)")
            continue
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    logger.info(f"\n{'='*80}")
    logger.info(f"PIPELINE COMPLETE!")
    logger.info(f"{'='*80}")
    
    logger.info(f"\nResults Summary:")
    logger.info(f"  Model: {model_output}")
    logger.info(f"  Evaluations: {eval_output}")
    
    if eval_results:
        logger.info(f"\nMoleculeNet Performance:")
        for dataset_name, metrics in eval_results.items():
            logger.info(f"\n  {dataset_name}:")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    logger.info(f"    {metric_name}: {metric_value:.4f}")
    
    # Save summary
    summary = {
        "model_path": str(model_output),
        "hp_search_path": str(hp_output) if not args.skip_hp_search else None,
        "evaluation_results": eval_results,
    }
    
    summary_file = output_dir / "pipeline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✓ Summary saved to {summary_file}")
    logger.info(f"\n🎉 All done! Check {output_dir} for results.")


if __name__ == "__main__":
    main()