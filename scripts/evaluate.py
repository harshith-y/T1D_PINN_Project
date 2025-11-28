#!/usr/bin/env python3
"""
Enhanced Evaluation Script v2

Comprehensive model evaluation with:
- Standard metrics (RMSE, MAE, MAPE)
- k-step ahead prediction (optional)
- Clinical zone analysis
- CSV export

Usage:
    # Basic evaluation
    python scripts/evaluate_v2.py --results-dir results/birnn_forward/Pat3_20241127_143022
    
    # With k-step evaluation
    python scripts/evaluate_v2.py --results-dir results/birnn_forward/Pat3_* --k-step 15
    
    # Batch evaluation
    python scripts/evaluate_v2.py --batch "results/birnn_forward/Pat*" --output batch_results.csv
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
from glob import glob

from src.training.predictions import PredictionManager
from src.evaluation.metrics import compute_glucose_metrics, compute_latent_metrics


def evaluate_single(results_dir: Path, k_step: int = None) -> dict:
    """
    Evaluate a single results directory.
    
    Args:
        results_dir: Path to results directory
        k_step: Optional k for k-step ahead evaluation
    
    Returns:
        Dictionary of metrics
    """
    print(f"\nEvaluating: {results_dir}")
    
    # Load predictions
    pred_mgr = PredictionManager(results_dir)
    
    if not pred_mgr.exists():
        print("   ❌ No predictions found")
        return {}
    
    predictions = pred_mgr.load()
    metadata = predictions.get('metadata', {})
    
    # Compute glucose metrics
    glucose_metrics = compute_glucose_metrics(
        pred=predictions['glucose_pred'],
        true=predictions['glucose_true'],
        split_idx=predictions.get('split_idx')
    )
    
    # Compute latent metrics if available
    latent_metrics = compute_latent_metrics(
        insulin_pred=predictions.get('insulin_pred'),
        insulin_true=predictions.get('insulin_true'),
        digestion_pred=predictions.get('digestion_pred'),
        digestion_true=predictions.get('digestion_true'),
        split_idx=predictions.get('split_idx')
    )
    
    # Combine all metrics
    all_metrics = {
        'results_dir': str(results_dir),
        'model': metadata.get('model_name', 'unknown'),
        'patient': metadata.get('patient', 'unknown'),
        **glucose_metrics,
        **latent_metrics
    }
    
    # Add parameter estimation metrics if available (inverse mode)
    for param_name in ['ksi', 'kl', 'ku_Vi']:
        param_data = pred_mgr.load_parameter_evolution(param_name)
        if param_data is not None:
            all_metrics[f'{param_name}_estimated'] = float(param_data['param_values'][-1])
            if 'true_value' in param_data:
                all_metrics[f'{param_name}_true'] = float(param_data['true_value'])
                all_metrics[f'{param_name}_error_percent'] = float(param_data['errors_percent'][-1])
    
    # k-step evaluation (if requested)
    if k_step is not None:
        print(f"   ⚠️  k-step evaluation not yet implemented")
        # TODO: Implement k-step evaluation
        # This would require loading the model, which we're avoiding for speed
    
    # Print summary
    print(f"   ✅ Glucose RMSE: {glucose_metrics['rmse']:.2f} mg/dL")
    if 'rmse_test' in glucose_metrics:
        print(f"      Test RMSE: {glucose_metrics['rmse_test']:.2f} mg/dL")
    
    if 'insulin_rmse' in latent_metrics:
        print(f"      Insulin RMSE: {latent_metrics['insulin_rmse']:.4f} U/dL")
    
    return all_metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Single results directory to evaluate')
    parser.add_argument('--batch', type=str, default=None,
                        help='Batch evaluation pattern (e.g., "results/birnn_forward/Pat*")')
    parser.add_argument('--k-step', type=int, default=None,
                        help='Optional k-step ahead evaluation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    if args.results_dir is None and args.batch is None:
        parser.error("Must specify either --results-dir or --batch")
    
    print("="*80)
    print("EVALUATION v2 (Enhanced)")
    print("="*80)
    
    # ========================================================================
    # Single Directory Evaluation
    # ========================================================================
    if args.results_dir:
        results_dir = Path(args.results_dir)
        
        if not results_dir.exists():
            print(f"❌ Error: Results directory not found: {results_dir}")
            sys.exit(1)
        
        metrics = evaluate_single(results_dir, k_step=args.k_step)
        
        # Save to CSV if requested
        if args.output:
            df = pd.DataFrame([metrics])
            df.to_csv(args.output, index=False)
            print(f"\n💾 Results saved to: {args.output}")
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE")
        print("="*80)
    
    # ========================================================================
    # Batch Evaluation
    # ========================================================================
    else:
        results_dirs = sorted(glob(args.batch))
        
        if not results_dirs:
            print(f"❌ Error: No results found matching: {args.batch}")
            sys.exit(1)
        
        print(f"Found {len(results_dirs)} results directories")
        print("="*80)
        
        all_metrics = []
        
        for results_dir in results_dirs:
            try:
                metrics = evaluate_single(Path(results_dir), k_step=args.k_step)
                if metrics:
                    all_metrics.append(metrics)
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        # Create DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Save to CSV
        output_file = args.output or 'batch_evaluation.csv'
        df.to_csv(output_file, index=False)
        
        print("\n" + "="*80)
        print("✅ BATCH EVALUATION COMPLETE")
        print("="*80)
        print(f"\n📊 Summary:")
        print(f"   Total evaluated: {len(all_metrics)}")
        print(f"   Mean RMSE: {df['rmse'].mean():.2f} mg/dL")
        print(f"   Std RMSE: {df['rmse'].std():.2f} mg/dL")
        
        print(f"\n💾 Results saved to: {output_file}")
        print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)