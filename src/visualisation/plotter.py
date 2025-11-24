"""
Visualization module for T1D PINN experiments.

This module provides automated plotting functions that match the style and
conventions from the research notebooks. All plots are production-ready with
high DPI and consistent styling.

Usage:
    from src.visualization.plotter import ExperimentPlotter
    
    plotter = ExperimentPlotter(save_dir="results/experiment_001/plots")
    plotter.plot_all(predictions, metrics, config)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set consistent style
sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 150  # High quality for saving
plt.rcParams['savefig.dpi'] = 600  # Publication quality


# ============================================================================
# MODEL COLOR SCHEME (from notebooks)
# ============================================================================

MODEL_COLORS = {
    "PINN": "#009E73",           # Green
    "Modified MLP": "#0072B2",   # Blue
    "BI-RNN": "#D55E00"          # Orange-red
}

MODEL_STYLES = {
    "PINN": {"color": "#009E73", "linestyle": "--", "linewidth": 2},
    "Modified MLP": {"color": "#0072B2", "linestyle": "--", "linewidth": 2},
    "BI-RNN": {"color": "#D55E00", "linestyle": "--", "linewidth": 2}
}


# ============================================================================
# CORE PLOTTING FUNCTIONS
# ============================================================================

class ExperimentPlotter:
    """
    Automated plotting for T1D PINN experiments.
    
    This class generates all standard plots for model evaluation:
    - Time series predictions (glucose, insulin, digestion)
    - Training diagnostics (loss curves, parameter evolution)
    - Statistical comparisons (RMSE, relative error)
    - Multi-model comparisons
    
    All plots match the style from research notebooks with consistent
    formatting, colors, and layouts.
    """
    
    def __init__(
        self,
        save_dir: str | Path,
        model_name: str = "Model",
        dpi: int = 600,
        figsize_default: Tuple[int, int] = (8, 5),
        figsize_timeseries: Tuple[int, int] = (8, 8)
    ):
        """
        Initialize plotter.
        
        Args:
            save_dir: Directory to save plots
            model_name: Name of model for titles
            dpi: DPI for saved figures (default 600 for publication)
            figsize_default: Default figure size for standard plots
            figsize_timeseries: Figure size for time series plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.dpi = dpi
        self.figsize_default = figsize_default
        self.figsize_timeseries = figsize_timeseries
        
        # Get model style if known
        self.style = MODEL_STYLES.get(model_name, {"color": "blue", "linestyle": "--", "linewidth": 2})
    
    def plot_glucose_prediction(
        self,
        time: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        split_idx: Optional[int] = None,
        title: Optional[str] = None,
        ylabel: str = "Normalized Glycaemia",
        save_name: str = "glucose_prediction.png"
    ) -> Path:
        """
        Plot glucose prediction vs ground truth with train/test split.
        
        This is the most important plot - shows model performance on
        glucose prediction with clear train/test boundary.
        
        Args:
            time: Time array (minutes)
            y_true: True glucose values
            y_pred: Predicted glucose values
            split_idx: Index of train/test split (draws vertical line)
            title: Plot title (auto-generated if None)
            ylabel: Y-axis label
            save_name: Filename for saved plot
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize_timeseries)
        
        # Plot data
        ax.plot(time, y_true, label="True Glucose", color="blue", linewidth=2)
        ax.plot(time, y_pred, label="Predicted Glucose", 
               linestyle="--", color="red", linewidth=2)
        
        # Train/test split line
        if split_idx is not None:
            split_time = time[split_idx]
            ax.axvline(x=split_time, color='black', linestyle='--', linewidth=1.5)
            
            # Add region labels
            mid_train = time[split_idx // 2]
            mid_test = time[split_idx + (len(time) - split_idx) // 2]
            y_top = ax.get_ylim()[1] * 0.96
            ax.text(mid_train, y_top, 'Training', ha='center', fontsize=11)
            ax.text(mid_test, y_top, 'Test', ha='center', fontsize=11)
        
        # Labels and formatting
        ax.set_xlabel("Time (minutes)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title or f"Glycaemia Prediction: {self.model_name}", fontsize=14)
        ax.legend(loc="lower left", frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        
        return save_path
    
    def plot_latent_variables(
        self,
        time: np.ndarray,
        I_true: np.ndarray,
        I_pred: np.ndarray,
        D_true: np.ndarray,
        D_pred: np.ndarray,
        split_idx: Optional[int] = None,
        save_name: str = "latent_variables.png"
    ) -> Path:
        """
        Plot insulin and digestion state predictions.
        
        Creates two subplots for I(t) and D(t) with train/test split.
        
        Args:
            time: Time array (minutes)
            I_true: True insulin values
            I_pred: Predicted insulin values
            D_true: True digestion values
            D_pred: Predicted digestion values
            split_idx: Index of train/test split
            save_name: Filename for saved plot
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize_timeseries[0], 10), sharex=True)
        
        # Insulin subplot
        ax1.plot(time, I_true, label="True Insulin", color="blue", linewidth=2)
        ax1.plot(time, I_pred, label="Predicted Insulin", 
                linestyle="--", color="red", linewidth=2)
        
        if split_idx is not None:
            ax1.axvline(x=time[split_idx], color='black', linestyle='--', linewidth=1.5)
        
        ax1.set_ylabel("Insulinemia (U/dL)", fontsize=12)
        ax1.set_title(f"Insulin Prediction: {self.model_name}", fontsize=14)
        ax1.legend(loc="best", fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Digestion subplot
        ax2.plot(time, D_true, label="True Digestion", color="blue", linewidth=2)
        ax2.plot(time, D_pred, label="Predicted Digestion", 
                linestyle="--", color="red", linewidth=2)
        
        if split_idx is not None:
            ax2.axvline(x=time[split_idx], color='black', linestyle='--', linewidth=1.5)
        
        ax2.set_xlabel("Time (minutes)", fontsize=12)
        ax2.set_ylabel("Digestion Rate (mg/dL/min)", fontsize=12)
        ax2.set_title(f"Digestion Prediction: {self.model_name}", fontsize=14)
        ax2.legend(loc="best", fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        
        return save_path
    
    def plot_loss_curves(
        self,
        history: Dict[str, List[float]],
        log_scale: bool = False,
        save_name: str = "loss_curves.png"
    ) -> Path:
        """
        Plot training loss evolution over epochs.
        
        Shows how different loss components evolve during training.
        
        Args:
            history: Dictionary with keys like 'loss', 'loss_g', 'loss_B', etc.
            log_scale: Whether to use log scale for y-axis
            save_name: Filename for saved plot
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize_default)
        
        # Plot each loss component
        for key, values in history.items():
            if len(values) > 0 and key.startswith('loss'):
                label = key.replace('loss_', '').replace('loss', 'total')
                ax.plot(values, label=label, linewidth=1.5, alpha=0.8)
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"Training Loss Evolution: {self.model_name}", fontsize=14)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        
        return save_path
    
    def plot_parameter_evolution(
        self,
        param_history: List[float],
        true_value: Optional[float] = None,
        param_name: str = "ksi",
        save_name: str = "parameter_evolution.png"
    ) -> Path:
        """
        Plot parameter convergence during inverse training.
        
        Critical for inverse problems - shows whether parameter
        estimation is converging to true value.
        
        Args:
            param_history: List of parameter values over epochs
            true_value: True parameter value (draws horizontal line)
            param_name: Parameter name for labels
            save_name: Filename for saved plot
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize_default)
        
        epochs = np.arange(len(param_history))
        ax.plot(epochs, param_history, label=f"Estimated {param_name}", 
               color="blue", linewidth=2)
        
        if true_value is not None:
            ax.axhline(y=true_value, color='red', linestyle='--', 
                      linewidth=2, label=f"True {param_name}")
            
            # Calculate and display final error
            final_val = param_history[-1]
            error = abs(final_val - true_value) / true_value * 100
            ax.text(0.98, 0.02, f"Final Error: {error:.2f}%", 
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10)
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(f"{param_name} Value", fontsize=12)
        ax.set_title(f"Parameter Convergence: {param_name}", fontsize=14)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        
        return save_path
    
    def plot_residuals(
        self,
        time: np.ndarray,
        residuals: np.ndarray,
        split_idx: Optional[int] = None,
        title: Optional[str] = None,
        save_name: str = "residuals.png"
    ) -> Path:
        """
        Plot prediction residuals (error over time).
        
        Helps identify where model struggles - e.g., sharp transitions,
        hypoglycemic events, post-meal spikes.
        
        Args:
            time: Time array (minutes)
            residuals: Prediction errors (y_pred - y_true)
            split_idx: Index of train/test split
            title: Plot title
            save_name: Filename for saved plot
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize_default)
        
        ax.plot(time, residuals, color="darkred", linewidth=1.5, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        if split_idx is not None:
            ax.axvline(x=time[split_idx], color='black', linestyle='--', linewidth=1.5)
        
        ax.set_xlabel("Time (minutes)", fontsize=12)
        ax.set_ylabel("Residual (Predicted - True)", fontsize=12)
        ax.set_title(title or f"Prediction Residuals: {self.model_name}", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        
        return save_path
    
    def plot_rmse_comparison(
        self,
        rmse_data: Dict[str, float],
        save_name: str = "rmse_comparison.png"
    ) -> Path:
        """
        Plot RMSE for different variables (G, I, D).
        
        Args:
            rmse_data: Dictionary like {'G': 18.05, 'I': 0.02, 'D': 1.5}
            save_name: Filename for saved plot
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        
        variables = list(rmse_data.keys())
        values = list(rmse_data.values())
        
        bars = ax.bar(variables, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel("RMSE", fontsize=12)
        ax.set_title(f"RMSE by Variable: {self.model_name}", fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        
        return save_path
    
    def plot_patient_data_only(
        self,
        time: np.ndarray,
        glucose: np.ndarray,
        insulin: Optional[np.ndarray] = None,
        digestion: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        r: Optional[np.ndarray] = None,
        patient_id: str = "Patient",
        save_name: str = "patient_data_profile.png"
    ) -> Path:
        """
        Plot raw patient data without predictions.
        
        Useful for exploratory data analysis - visualize glucose, insulin,
        digestion states and inputs (u, r) for a patient without any model.
        
        Args:
            time: Time array (minutes)
            glucose: Glucose measurements (mg/dL)
            insulin: Optional insulin state I(t)
            digestion: Optional digestion state D(t)
            u: Optional insulin inputs (U/min)
            r: Optional carb inputs (g/min)
            patient_id: Patient identifier for title
            save_name: Filename for saved plot
            
        Returns:
            Path to saved plot
            
        Example:
            >>> from src.datasets.loader import load_synthetic_window
            >>> data = load_synthetic_window(patient=3)
            >>> plotter = ExperimentPlotter(save_dir="results/eda")
            >>> plotter.plot_patient_data_only(
            ...     time=data.t_min,
            ...     glucose=data.glucose,
            ...     insulin=data.insulin,
            ...     digestion=data.digestion,
            ...     u=data.u,
            ...     r=data.r,
            ...     patient_id="Pat3"
            ... )
        """
        # Count how many subplots we need
        n_plots = 1  # Always have glucose
        if insulin is not None:
            n_plots += 1
        if digestion is not None:
            n_plots += 1
        if u is not None or r is not None:
            n_plots += 1
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
        
        # Make axes always a list for consistent indexing
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # 1. Glucose (always present)
        axes[plot_idx].plot(time, glucose, linewidth=2, color='#1f77b4', label='Glucose')
        axes[plot_idx].set_ylabel('Glucose (mg/dL)', fontsize=12)
        axes[plot_idx].set_title(f'{patient_id} - Data Profile', fontsize=14, fontweight='bold')
        axes[plot_idx].legend(loc='best', fontsize=10)
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
        
        # 2. Insulin (if available)
        if insulin is not None:
            axes[plot_idx].plot(time, insulin, linewidth=2, color='#2ca02c', label='Insulin')
            axes[plot_idx].set_ylabel('Insulinemia I(t) (U/dL)', fontsize=12)
            axes[plot_idx].legend(loc='best', fontsize=10)
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # 3. Digestion (if available)
        if digestion is not None:
            axes[plot_idx].plot(time, digestion, linewidth=2, color='#ff7f0e', label='Digestion')
            axes[plot_idx].set_ylabel('Digestion D(t) (mg/dL/min)', fontsize=12)
            axes[plot_idx].legend(loc='best', fontsize=10)
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # 4. Inputs u(t) and r(t) (if available)
        if u is not None or r is not None:
            ax = axes[plot_idx]
            if u is not None:
                ax.plot(time, u, linewidth=1.5, color='#d62728', label='Insulin Input u(t)', alpha=0.7)
            if r is not None:
                # Use secondary y-axis for carbs (different scale)
                if u is not None:
                    ax2 = ax.twinx()
                    ax2.plot(time, r, linewidth=1.5, color='#9467bd', label='Carb Input r(t)', alpha=0.7)
                    ax2.set_ylabel('Carbs r(t) (g/min)', fontsize=12, color='#9467bd')
                    ax2.tick_params(axis='y', labelcolor='#9467bd')
                    ax2.legend(loc='upper right', fontsize=10)
                else:
                    ax.plot(time, r, linewidth=1.5, color='#9467bd', label='Carb Input r(t)', alpha=0.7)
            
            ax.set_ylabel('Insulin u(t) (U/min)', fontsize=12, color='#d62728' if u is not None else 'black')
            if u is not None:
                ax.tick_params(axis='y', labelcolor='#d62728')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # X-axis label on bottom plot
        axes[-1].set_xlabel('Time (minutes)', fontsize=12)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_all(
        self,
        predictions: Dict[str, np.ndarray],
        ground_truth: Dict[str, np.ndarray],
        metrics: Dict[str, float],
        history: Optional[Dict[str, List[float]]] = None,
        param_history: Optional[List[float]] = None,
        true_param: Optional[float] = None,
        param_name: str = "ksi",
        split_idx: Optional[int] = None
    ) -> Dict[str, Path]:
        """
        Generate all standard plots for an experiment.
        
        This is the main function to call - creates complete set of
        diagnostic plots for model evaluation.
        
        Args:
            predictions: Dict with keys 'glucose', 'insulin', 'digestion', 'time'
            ground_truth: Dict with same keys as predictions
            metrics: Dict with RMSE values
            history: Training history (optional)
            param_history: Parameter evolution (optional, for inverse)
            true_param: True parameter value (optional, for inverse)
            param_name: Name of inverse parameter
            split_idx: Train/test split index
            
        Returns:
            Dictionary mapping plot names to saved paths
        """
        saved_plots = {}
        
        # 1. Glucose prediction (ALWAYS)
        if 'glucose' in predictions and 'time' in predictions:
            path = self.plot_glucose_prediction(
                time=predictions['time'],
                y_true=ground_truth['glucose'],
                y_pred=predictions['glucose'],
                split_idx=split_idx
            )
            saved_plots['glucose_prediction'] = path
            print(f"✅ Saved: {path.name}")
            
            # Residuals
            residuals = predictions['glucose'] - ground_truth['glucose']
            path = self.plot_residuals(
                time=predictions['time'],
                residuals=residuals,
                split_idx=split_idx
            )
            saved_plots['residuals'] = path
            print(f"✅ Saved: {path.name}")
        
        # 2. Latent variables (if available)
        if all(k in predictions for k in ['insulin', 'digestion']):
            path = self.plot_latent_variables(
                time=predictions['time'],
                I_true=ground_truth['insulin'],
                I_pred=predictions['insulin'],
                D_true=ground_truth['digestion'],
                D_pred=predictions['digestion'],
                split_idx=split_idx
            )
            saved_plots['latent_variables'] = path
            print(f"✅ Saved: {path.name}")
        
        # 3. Loss curves (if history provided)
        if history is not None and len(history.get('loss', [])) > 0:
            path = self.plot_loss_curves(history)
            saved_plots['loss_curves'] = path
            print(f"✅ Saved: {path.name}")
        
        # 4. Parameter evolution (if inverse training)
        if param_history is not None and len(param_history) > 0:
            path = self.plot_parameter_evolution(
                param_history=param_history,
                true_value=true_param,
                param_name=param_name
            )
            saved_plots['parameter_evolution'] = path
            print(f"✅ Saved: {path.name}")
        
        # 5. RMSE comparison
        if metrics:
            rmse_data = {
                k.replace('rmse_', '').upper(): v 
                for k, v in metrics.items() 
                if 'rmse' in k.lower()
            }
            if rmse_data:
                path = self.plot_rmse_comparison(rmse_data)
                saved_plots['rmse_comparison'] = path
                print(f"✅ Saved: {path.name}")
        
        return saved_plots


# ============================================================================
# MULTI-MODEL COMPARISON FUNCTIONS
# ============================================================================

def plot_multi_model_glucose(
    results: Dict[str, Dict[str, np.ndarray]],
    ground_truth: Dict[str, np.ndarray],
    save_path: str | Path,
    split_idx: Optional[int] = None,
    dpi: int = 600
) -> Path:
    """
    Plot glucose predictions from multiple models on same plot.
    
    Args:
        results: Dict mapping model names to prediction dicts
                 e.g., {"PINN": {...}, "Modified MLP": {...}, "BI-RNN": {...}}
        ground_truth: Dict with 'time' and 'glucose' keys
        save_path: Path to save plot
        split_idx: Train/test split index
        dpi: DPI for saved figure
        
    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time = ground_truth['time']
    
    # Plot ground truth
    ax.plot(time, ground_truth['glucose'], 
           label="True Glucose", color="blue", linewidth=2)
    
    # Plot each model
    for model_name, preds in results.items():
        style = MODEL_STYLES.get(model_name, {"color": "gray", "linestyle": "--", "linewidth": 2})
        ax.plot(time, preds['glucose'], 
               label=f"{model_name}", **style)
    
    # Train/test split
    if split_idx is not None:
        ax.axvline(x=time[split_idx], color='black', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Normalized Glycaemia", fontsize=12)
    ax.set_title("Multi-Model Glucose Prediction Comparison", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_path)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    return save_path


def plot_rmse_boxplot_multi_patient(
    results_df: pd.DataFrame,
    metric: str = "G_RMSE",
    save_path: str | Path = "rmse_boxplot.png",
    dpi: int = 600
) -> Path:
    """
    Create boxplot comparing RMSE across patients and models.
    
    Args:
        results_df: DataFrame with columns ['Patient', 'Model', metric]
        metric: Metric to plot (e.g., 'G_RMSE', 'I_RMSE')
        save_path: Path to save plot
        dpi: DPI for saved figure
        
    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(data=results_df, x="Model", y=metric, palette="Set2", ax=ax)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"{metric} Distribution Across Patients", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = Path(save_path)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    return save_path


def plot_ksi_error_distribution(
    results_df: pd.DataFrame,
    save_path: str | Path = "ksi_error_distribution.png",
    dpi: int = 600
) -> Path:
    """
    Plot distribution of relative error in ksi estimation across models.
    
    Args:
        results_df: DataFrame with columns ['Model', 'Rel_Error']
        save_path: Path to save plot
        dpi: DPI for saved figure
        
    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.boxplot(data=results_df, x="Model", y="Rel_Error", palette="Set2", ax=ax)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Relative Error (%)", fontsize=12)
    ax.set_title("Distribution of Relative Error in ksi Estimation", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = Path(save_path)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    return save_path


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_results_summary_table(
    results: Dict[str, Dict[str, Any]],
    save_path: str | Path = "results_summary.csv"
) -> Path:
    """
    Create a summary table of results for all models.
    
    Args:
        results: Dict mapping model names to result dicts
        save_path: Path to save CSV
        
    Returns:
        Path to saved CSV
    """
    rows = []
    for model_name, res in results.items():
        row = {'Model': model_name}
        if 'metrics' in res:
            row.update(res['metrics'])
        if 'ksi_error' in res:
            row['ksi_error_pct'] = res['ksi_error']
        rows.append(row)
    
    df = pd.DataFrame(rows)
    save_path = Path(save_path)
    df.to_csv(save_path, index=False)
    
    return save_path


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    
    # Fix imports: Add project root to sys.path
    # This allows "from src.datasets.loader import ..." to work
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    parser = argparse.ArgumentParser(
        description="Test visualization pipeline with real or dummy data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with dummy data
  python plotter.py
  
  # Test with synthetic patient data
  python plotter.py --patient 3 --data-dir data/synthetic
  
  # Test with REAL patient data (NEW!)
  python plotter.py --real-patient 3
  
  # Plot ONLY patient data (no predictions) - for EDA
  python plotter.py --patient 3 --data-only
  python plotter.py --real-patient 5 --data-only
  
  # Custom save directory
  python plotter.py --real-patient 3 --save-dir results/real_pat3_viz
"""
    )
    
    parser.add_argument(
        '--patient', 
        type=int, 
        default=None,
        help='Synthetic patient number (2-11) to load. Use with --data-dir.'
    )
    parser.add_argument(
        '--real-patient',
        type=int,
        default=None,
        help='Real patient number (1-15) to load from data/processed/RealPat{N}.csv'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/synthetic',
        help='Directory containing Pat{N}.csv files (default: data/synthetic)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='results/test_plots',  # Changed from /tmp/test_plots
        help='Directory to save test plots (default: results/test_plots)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='Test Model',
        help='Model name for plot titles (default: Test Model)'
    )
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Plot only patient data (no predictions). Only works with --patient.'
    )
    
    args = parser.parse_args()
    
    # Flag to track if we should skip prediction generation
    skip_predictions = False
    
    print("="*80)
    print("VISUALIZATION PIPELINE TEST")
    print("="*80)
    
    # Check for conflicting arguments
    if args.patient is not None and args.real_patient is not None:
        print("\n❌ Error: Cannot use both --patient and --real-patient at the same time!")
        print("   Use --patient for synthetic data (Pat2-11)")
        print("   Use --real-patient for real data (RealPat1-15)")
        sys.exit(1)
    
    if args.real_patient is not None:
        # Load REAL patient data
        print(f"\n📁 Loading REAL patient data: RealPat{args.real_patient}")
        try:
            # TODO: Once load_real_patient_csv() is added to loader.py, replace with:
            # from src.datasets.loader import load_real_patient_csv
            # data = load_real_patient_csv(patient=args.real_patient)
            
            # Current: Load CSV directly (loader.py's load_real_window expects JSON)
            csv_path = Path('data/processed') / f'RealPat{args.real_patient}.csv'
            
            if not csv_path.exists():
                raise FileNotFoundError(f"File not found: {csv_path}")
            
            print(f"✅ Loading from: {csv_path}")
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Check what columns we have
            print(f"   CSV columns: {list(df.columns)}")
            
            # Extract data (flexible about what columns exist)
            required_cols = ['time', 'glucose']
            missing = set(required_cols) - set(df.columns)
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")
            
            # Extract arrays
            time = df['time'].values.astype(np.float32)
            glucose = df['glucose'].values.astype(np.float32)
            
            # Optional columns
            insulin = df['insulin'].values.astype(np.float32) if 'insulin' in df.columns else None
            digestion = df['carbohydrates'].values.astype(np.float32) if 'carbohydrates' in df.columns else None
            u = df['u'].values.astype(np.float32) if 'u' in df.columns else None
            r = df['r'].values.astype(np.float32) if 'r' in df.columns else None
            
            # Create a simple data container
            class SimpleData:
                def __init__(self):
                    self.t_min = time - time[0]  # Start from 0
                    self.glucose = glucose
                    self.insulin = insulin
                    self.digestion = digestion
                    self.u = u
                    self.r = r
                    self.patient_id = f"RealPat{args.real_patient}"
                    self.has_latent_states = (insulin is not None and digestion is not None)
            
            data = SimpleData()
            
            print(f"✅ Data loaded successfully!")
            print(f"   Patient: {data.patient_id}")
            print(f"   Data points: {len(time)}")
            print(f"   Glucose range: {glucose.min():.1f} - {glucose.max():.1f} mg/dL")
            print(f"   Duration: {time[-1] - time[0]:.1f} minutes")
            print(f"   Has latent states: {data.has_latent_states}")
                
            # Check if user wants data-only plot
            if args.data_only:
                    print(f"\n📊 Creating data-only plot (no predictions)...")
                    
                    plotter = ExperimentPlotter(
                        save_dir=args.save_dir,
                        model_name=args.model_name
                    )
                    
                    path = plotter.plot_patient_data_only(
                        time=data.t_min,
                        glucose=data.glucose,
                        insulin=data.insulin if data.has_latent_states else None,
                        digestion=data.digestion if data.has_latent_states else None,
                        u=data.u if hasattr(data, 'u') else None,
                        r=data.r if hasattr(data, 'r') else None,
                        patient_id=data.patient_id,
                        save_name=f"{data.patient_id}_profile.png"
                    )
                    
                    print("\n" + "="*80)
                    print("✅ REAL PATIENT DATA-ONLY PLOT COMPLETE!")
                    print("="*80)
                    print(f"\nGenerated plot:")
                    print(f"   📈 {data.patient_id}_profile.png: {path}")
                    print(f"\n💡 View plot with:")
                    print(f"   open {path}")
                    
                    skip_predictions = True  # Don't generate fake predictions
            
            else:
                # Use real data with fake predictions for testing (non-data-only mode)
                time = data.t_min
                predictions = {
                    'time': time,
                    'glucose': data.glucose + np.random.randn(len(time)) * 3,  # Add small noise
                }
                
                # Add latent variables if available
                if data.has_latent_states:
                    predictions['insulin'] = data.insulin
                    predictions['digestion'] = data.digestion
                
                ground_truth = {
                    'glucose': data.glucose,
                }
                
                if data.has_latent_states:
                    ground_truth['insulin'] = data.insulin
                    ground_truth['digestion'] = data.digestion
                
                metrics = {
                    'rmse_glucose': 18.5,
                    'rmse_total': 19.2
                }
                
                if data.has_latent_states:
                    metrics['rmse_insulin'] = 0.03
                    metrics['rmse_digestion'] = 1.5
                
                # Simulate parameter history (for inverse mode demo)
                param_history = np.linspace(265, 270, 100).tolist()
                true_param = 270.0
                
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print(f"   Looking for: data/processed/RealPat{args.real_patient}.csv")
            print("   Falling back to dummy data...\n")
            args.real_patient = None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            print("   Falling back to dummy data...\n")
            args.real_patient = None
    
    if args.patient is not None:
        # Load real patient data
        print(f"\n📁 Loading real data: Patient {args.patient}")
        try:
            # Import loader (adjust path if needed)
            try:
                from src.datasets.loader import load_synthetic_window
            except ImportError as e:
                print(f"⚠️  Import error: {e}")
                print("   Troubleshooting:")
                print(f"   - Current directory: {Path.cwd()}")
                print(f"   - Project root added to path: {project_root}")
                print(f"   - Does src/ exist? {(project_root / 'src').exists()}")
                print(f"   - Does src/datasets/ exist? {(project_root / 'src' / 'datasets').exists()}")
                print("   Falling back to dummy data...\n")
                args.patient = None
            
            if args.patient is not None:
                data = load_synthetic_window(patient=args.patient, root=args.data_dir)
                
                print(f"✅ Data loaded: {data}")
                
                # Check if user wants data-only plot
                if args.data_only:
                    print(f"\n📊 Creating data-only plot (no predictions)...")
                    
                    plotter = ExperimentPlotter(
                        save_dir=args.save_dir,
                        model_name=args.model_name
                    )
                    
                    path = plotter.plot_patient_data_only(
                        time=data.t_min,
                        glucose=data.glucose,
                        insulin=data.insulin if data.has_latent_states else None,
                        digestion=data.digestion if data.has_latent_states else None,
                        u=data.u,
                        r=data.r,
                        patient_id=data.patient_id
                    )
                    
                    print("\n" + "="*80)
                    print("✅ DATA-ONLY PLOT COMPLETE!")
                    print("="*80)
                    print(f"\nGenerated plot:")
                    print(f"   📈 patient_data_profile.png: {path}")
                    print(f"\n💡 View plot with:")
                    print(f"   open {path}")
                    
                    skip_predictions = True  # Don't generate fake predictions
                
                # Use real data with fake predictions for testing
                time = data.t_min
                predictions = {
                    'time': time,
                    'glucose': data.glucose + np.random.randn(len(time)) * 2,  # Add small noise
                    'insulin': data.insulin,
                    'digestion': data.digestion
                }
                ground_truth = {
                    'glucose': data.glucose,
                    'insulin': data.insulin,
                    'digestion': data.digestion
                }
                metrics = {
                    'rmse_glucose': 15.5,
                    'rmse_insulin': 0.02,
                    'rmse_digestion': 1.2,
                    'rmse_total': 16.8
                }
                
                # Simulate parameter history (for inverse mode demo)
                param_history = np.linspace(270, 274, 100).tolist()
                true_param = 274.0
                
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print("   Falling back to dummy data...\n")
            args.patient = None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("   Falling back to dummy data...\n")
            args.patient = None
    
    if args.patient is None and args.real_patient is None:
        # Generate dummy data
        print("\n🎲 Generating dummy test data...")
        time = np.linspace(0, 2880, 2881)
        y_true = 100 + 20 * np.sin(2 * np.pi * time / 1440) + np.random.randn(len(time)) * 5
        y_pred = y_true + np.random.randn(len(time)) * 10
        
        # Check if data-only mode
        if args.data_only:
            print(f"\n📊 Creating data-only plot (no predictions)...")
            
            plotter = ExperimentPlotter(
                save_dir=args.save_dir,
                model_name=args.model_name
            )
            
            path = plotter.plot_patient_data_only(
                time=time,
                glucose=y_true,  # Use ground truth only
                insulin=None,
                digestion=None,
                u=None,
                r=None,
                patient_id="Dummy Data"
            )
            
            print("\n" + "="*80)
            print("✅ DATA-ONLY PLOT COMPLETE!")
            print("="*80)
            print(f"\nGenerated plot:")
            print(f"   📈 patient_data_profile.png: {path}")
            print(f"\n💡 View plot with:")
            print(f"   open {path}")
            
            skip_predictions = True
        else:
            # Normal test mode with predictions
            predictions = {
                'time': time,
                'glucose': y_pred
            }
            ground_truth = {
                'glucose': y_true
            }
            metrics = {
                'rmse_glucose': 10.5,
                'rmse_total': 12.3
            }
            param_history = None
            true_param = None
    
    # Only generate prediction plots if not in data-only mode
    if not skip_predictions:
        # Create plotter
        print(f"\n📊 Creating plotter...")
        print(f"   Save directory: {args.save_dir}")
        print(f"   Model name: {args.model_name}")
        
        plotter = ExperimentPlotter(
            save_dir=args.save_dir, 
            model_name=args.model_name
        )
        
        # Generate plots
        print(f"\n🎨 Generating plots...")
        split_idx = int(0.9 * len(predictions['time']))
        
        paths = plotter.plot_all(
            predictions=predictions,
            ground_truth=ground_truth,
            metrics=metrics,
            param_history=param_history,
            true_param=true_param,
            param_name="ksi" if param_history else None,
            split_idx=split_idx
        )
        
        # Summary
        print("\n" + "="*80)
        print("✅ TEST COMPLETE!")
        print("="*80)
        print(f"\nGenerated {len(paths)} plots:")
        for name, path in paths.items():
            print(f"   📈 {name}: {path}")
        
        print(f"\n💡 View plots with:")
        print(f"   open {args.save_dir}")
        print(f"   # or on Linux: xdg-open {args.save_dir}")
        print(f"   # or on Windows: explorer {args.save_dir}")
        
        print("\n✨ Success! Visualization pipeline is working.")