#!/usr/bin/env python3
"""
Visualization Module for Covariate Study

This module contains all plotting and visualization functions for the covariate study,
including summary reports and prediction visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import logging
from pathlib import Path
from .eval import StudyResult


logger = logging.getLogger(__name__)


def create_summary_report(study_results: StudyResult, output_file: Path):
    """Create a summary report with statistics and comparisons."""

    logger.info(f"Creating summary report, output file: {output_file}")

    experiments = study_results.exp_results
    n_samples = study_results.n_samples
    covariate_types = study_results.covariate_types

    # Extract metrics for all experiments
    mase_no_cov = [exp.results_without_covariate.mase for exp in experiments]
    mase_with_cov = [exp.results_with_covariate.mase for exp in experiments]
    sql_no_cov = [exp.results_without_covariate.sql for exp in experiments]
    sql_with_cov = [exp.results_with_covariate.sql for exp in experiments]

    with PdfPages(output_file) as pdf:
        # Create summary page with 2x2 grid for two metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        # MASE comparison (top left)
        axes[0, 0].boxplot(
            [mase_no_cov, mase_with_cov], labels=["Without Covariate", "With Covariate"]
        )
        axes[0, 0].set_title("MASE Comparison Across All Experiments")
        axes[0, 0].set_ylabel("Mean Absolute Scaled Error")

        # SQL comparison (bottom left)
        axes[1, 0].boxplot(
            [sql_no_cov, sql_with_cov], labels=["Without Covariate", "With Covariate"]
        )
        axes[1, 0].set_title("SQL Comparison Across All Experiments")
        axes[1, 0].set_ylabel("Scaled Quantile Loss")

        # MASE improvement distribution (top right)
        mase_improvements = [
            ((no_cov - with_cov) / no_cov * 100)
            for no_cov, with_cov in zip(mase_no_cov, mase_with_cov)
        ]
        axes[0, 1].hist(
            mase_improvements, bins=10, alpha=0.7, edgecolor="black", color="lightgreen"
        )
        axes[0, 1].set_title("Distribution of MASE Improvements (%)")
        axes[0, 1].set_xlabel("Improvement (%)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].axvline(
            np.mean(mase_improvements),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(mase_improvements):.2f}%",
        )
        axes[0, 1].legend()

        # SQL improvement distribution (bottom right)
        sql_improvements = [
            ((no_cov - with_cov) / no_cov * 100)
            for no_cov, with_cov in zip(sql_no_cov, sql_with_cov)
        ]
        axes[1, 1].hist(
            sql_improvements, bins=10, alpha=0.7, edgecolor="black", color="lightsalmon"
        )
        axes[1, 1].set_title("Distribution of SQL Improvements (%)")
        axes[1, 1].set_xlabel("Improvement (%)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].axvline(
            np.mean(sql_improvements),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(sql_improvements):.2f}%",
        )
        axes[1, 1].legend()

        fig.suptitle(
            f"Covariate Study Summary: {', '.join(covariate_types).replace('_', ' ').title()}",
            fontsize=16,
            fontweight="bold",
            y=1.02,
            ha="center",
        )
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Create detailed statistics page
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Summary statistics text
        summary_text = f"""Covariate Study Detailed Results
Covariate Types: {", ".join(covariate_types)}
Number of Experiments: {n_samples}

MASE Results:
- Without Covariate: {np.mean(mase_no_cov):.4f} ± {np.std(mase_no_cov):.4f}
- With Covariate: {np.mean(mase_with_cov):.4f} ± {np.std(mase_with_cov):.4f}
- Average Improvement: {np.mean(mase_improvements):.2f}%
- Median Improvement: {np.median(mase_improvements):.2f}%

SQL Results:
- Without Covariate: {np.mean(sql_no_cov):.4f} ± {np.std(sql_no_cov):.4f}
- With Covariate: {np.mean(sql_with_cov):.4f} ± {np.std(sql_with_cov):.4f}
- Average Improvement: {np.mean(sql_improvements):.2f}%
- Median Improvement: {np.median(sql_improvements):.2f}%

Study Parameters:
- Weight: {study_results.parameters["weight"]}
- Relation: {study_results.parameters["relation"]}
- Train Ratio: {study_results.parameters["train_ratio"]}
- Covariate Weights: Randomized per experiment (Dirichlet distribution)

Best Performing Experiment (MASE):
- Experiment {np.argmax(mase_improvements) + 1}: {max(mase_improvements):.2f}% improvement

Worst Performing Experiment (MASE):
- Experiment {np.argmin(mase_improvements) + 1}: {min(mase_improvements):.2f}% improvement"""

        ax.text(
            0.05,
            0.95,
            summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Detailed Statistics", fontsize=14, pad=20)

        pdf.savefig(fig)
        plt.close()

    logger.info(f"Summary report saved to: {output_file}")


def format_config_dict(config: dict) -> str:
    """Format a dictionary of parameters into a string."""
    formatted_items = []
    for k, v in config.items():
        if isinstance(v, float):
            formatted_items.append(f"{k}: {v:.3f}")
        elif isinstance(v, tuple) and all(isinstance(x, float) for x in v):
            formatted_tuple = "(" + ", ".join(f"{x:.3f}" for x in v) + ")"
            formatted_items.append(f"{k}: {formatted_tuple}")
        elif isinstance(v, list) and all(isinstance(x, float) for x in v):
            formatted_list = "[" + ", ".join(f"{x:.3f}" for x in v) + "]"
            formatted_items.append(f"{k}: {formatted_list}")
        else:
            formatted_items.append(f"{k}: {v}")
    return ", ".join(formatted_items)


def create_prediction_visualizations(study_results: StudyResult, output_file: Path):
    """Create prediction visualization plots."""

    logger.info(f"Creating prediction visualizations, output file: {output_file}")

    experiments = study_results.exp_results
    n_samples = study_results.n_samples
    covariate_types = study_results.covariate_types

    with PdfPages(output_file) as pdf:
        # Individual experiment pages (8 experiments per page in 4x2 layout)
        n_detailed = min(16, n_samples)  # Show up to 16 experiments
        experiments_per_page = 8  # 8 experiments per page (4 rows x 2 columns)

        for page_start in range(0, n_detailed, experiments_per_page):
            page_end = min(page_start + experiments_per_page, n_detailed)

            # Create main figure with GridSpec for experiment layout (4 rows x 2 columns)
            fig = plt.figure(figsize=(16, 24))
            main_gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.25)

            for exp_idx, exp_num in enumerate(range(page_start, page_end)):
                exp_row = exp_idx // 2
                exp_col = exp_idx % 2

                # Create a nested GridSpec for this experiment (2 rows: predictions + covariate)
                exp_gs = main_gs[exp_row, exp_col].subgridspec(2, 1, hspace=0.05)

                exp = experiments[exp_num]

                x = np.arange(len(exp.data_with_covariate))
                n_train = exp.n_train_timesteps

                # Get predictions and quantiles
                pred_no_cov = exp.results_without_covariate.predictions
                quantiles_no_cov = exp.results_without_covariate.quantiles
                pred_with_cov = exp.results_with_covariate.predictions
                quantiles_with_cov = exp.results_with_covariate.quantiles

                # Predictions subplot
                ax_pred = fig.add_subplot(exp_gs[0])
                ax_pred.plot(
                    x,
                    exp.data_with_covariate,
                    label="Ground Truth",
                    color="blue",
                    linewidth=1.5,
                )
                ax_pred.plot(
                    x[n_train:],
                    pred_no_cov,
                    label="Pred w/o cov",
                    color="gray",
                    linewidth=1.5,
                    linestyle="--",
                )
                ax_pred.plot(
                    x[n_train:],
                    pred_with_cov,
                    label="Pred w/ cov",
                    color="coral",
                    linewidth=1.5,
                )

                # Add uncertainty bands for test period only
                test_x = x[n_train:]
                ax_pred.fill_between(
                    test_x,
                    quantiles_no_cov[0],
                    quantiles_no_cov[-1],
                    alpha=0.2,
                    color="gray",
                )
                ax_pred.fill_between(
                    test_x,
                    quantiles_with_cov[0],
                    quantiles_with_cov[-1],
                    alpha=0.2,
                    color="coral",
                )

                ax_pred.axvline(
                    x=n_train,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label="Train/Test Split",
                )

                # Calculate improvements
                mase_improvement = (
                    (
                        exp.results_without_covariate.mase
                        - exp.results_with_covariate.mase
                    )
                    / exp.results_without_covariate.mase
                    * 100
                )
                sql_improvement = (
                    (exp.results_without_covariate.sql - exp.results_with_covariate.sql)
                    / exp.results_without_covariate.sql
                    * 100
                )

                config_summary = [
                    (cov, f"{weight_str:.2f}")
                    for cov, weight_str in zip(
                        exp.covariate_types, exp.covariate_weights
                    )
                ]

                ax_pred.set_title(
                    f"Sample {exp_num + 1}\n"
                    f"{config_summary}\n"
                    f"MASE: {exp.results_without_covariate.mase:.3f} → "
                    f"{exp.results_with_covariate.mase:.3f} ({mase_improvement:+.1f}%)\n"
                    f"SQL: {exp.results_without_covariate.sql:.3f} → "
                    f"{exp.results_with_covariate.sql:.3f} ({sql_improvement:+.1f}%)",
                    fontsize=12,
                    fontweight="semibold",
                )
                ax_pred.legend(fontsize=8, loc="upper left")
                ax_pred.grid(True, alpha=0.3)
                ax_pred.set_ylabel("Value", fontsize=10)
                ax_pred.tick_params(labelsize=8)
                ax_pred.tick_params(labelbottom=False)

                # Covariate subplot (shares x-axis with predictions)
                ax_cov = fig.add_subplot(exp_gs[1], sharex=ax_pred)
                for i, cov_signal in enumerate(exp.individual_covariates):
                    label = (
                        f"{exp.covariate_types[i]} (w: {exp.covariate_weights[i]:.2f})"
                    )
                    ax_cov.plot(x, cov_signal, linewidth=1.5, label=label)

                ax_cov.axvline(
                    x=n_train,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label="Train/Test Split",
                )
                ax_cov.legend(fontsize=8, loc="upper left")
                ax_cov.grid(True, alpha=0.3)
                ax_cov.set_ylabel("Covariate", fontsize=10)
                ax_cov.set_xlabel("Time", fontsize=10)
                ax_cov.tick_params(labelsize=8)

            plt.suptitle(
                f"Prediction Visualizations: {', '.join(covariate_types).replace('_', ' ').title()} "
                f"(Page {page_start // experiments_per_page + 1})",
                fontsize=24,
                fontweight="bold",
            )
            pdf.savefig(fig)
            plt.close()

    logger.info(f"Prediction visualizations saved to: {output_file}")
