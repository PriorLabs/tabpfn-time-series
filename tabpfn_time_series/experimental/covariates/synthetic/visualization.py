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
from typing import List

from .eval import StudyResult, SingleExperimentResult


logger = logging.getLogger(__name__)


GROUND_TRUTH_COLOR = "blue"
PRED_W_COV_COLOR = "red"
PRED_WO_COV_COLOR = "green"
PRED_WO_COV_BAND_COLOR = "lightgreen"
TRAIN_TEST_SPLIT_COLOR = "gray"


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


def create_single_prediction_visualization(
    exp: SingleExperimentResult,
    show_legend: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create a single prediction visualization for an experiment, showing both predictions and covariates."""
    fig = plt.figure(figsize=(12, 4))

    gs = GridSpec(2, 1, figure=fig, hspace=0.05, height_ratios=[2, 1])
    ax_pred = fig.add_subplot(gs[0])
    ax_cov = fig.add_subplot(gs[1], sharex=ax_pred)

    plot_single_experiment(exp, ax_pred, ax_cov, show_legend)

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # fig.suptitle(f"Experiment {exp_num + 1} Visualization", fontsize=16, fontweight="bold")

    return fig, (ax_pred, ax_cov)


def create_main_prediction_visualization(
    exp_results: List[SingleExperimentResult],
    fontsize: float = 20,
    fontweight: str = "semibold",
    linewidth: float = 3,
    output_file: Path = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a main prediction visualization for an experiment, showing both predictions and covariates."""

    num_experiments = len(exp_results)

    fig = plt.figure(figsize=(10 * num_experiments, 5))
    gs = GridSpec(
        2, num_experiments, figure=fig, hspace=0.10, height_ratios=[3, 1], wspace=0.08
    )

    for exp_idx, exp in enumerate(exp_results):
        ax_pred = fig.add_subplot(gs[0, exp_idx])
        ax_cov = fig.add_subplot(gs[1, exp_idx], sharex=ax_pred)

        plot_single_experiment(
            exp,
            ax_pred,
            ax_cov,
            show_legend=False,
            fontsize=fontsize,
            fontweight=fontweight,
            linewidth=linewidth,
            use_xlabel=False,
            use_ylabel=True if exp_idx == 0 else False,
        )

    # Add a legend to the bottom of the figure
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=GROUND_TRUTH_COLOR,
            linewidth=linewidth,
            alpha=0.7,
            label="Ground Truth",
        ),
        plt.Line2D(
            [0], [0], color=PRED_W_COV_COLOR, linewidth=linewidth, label="Pred w/ cov."
        ),
        plt.Line2D(
            [0],
            [0],
            color=PRED_WO_COV_COLOR,
            linewidth=linewidth,
            label="Pred w/o cov.",
        ),
        plt.Line2D(
            [0],
            [0],
            color=TRAIN_TEST_SPLIT_COLOR,
            linewidth=linewidth,
            label="Train/Test Split",
            linestyle="--",
        ),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=fontsize,
        bbox_to_anchor=(0.5, -0.10),
    )
    legend.get_texts()[0].set_fontweight(fontweight)
    for text in legend.get_texts():
        text.set_fontweight(fontweight)

    if output_file:
        fig.savefig(output_file, bbox_inches="tight")
        plt.close()

    return fig, gs


def calculate_improvement(exp: SingleExperimentResult) -> tuple[float, float]:
    """Calculate the improvement in MASE and SQL for an experiment."""
    mase_improvement = (
        (exp.results_without_covariate.mase - exp.results_with_covariate.mase)
        / exp.results_without_covariate.mase
        * 100
    )
    sql_improvement = (
        (exp.results_without_covariate.sql - exp.results_with_covariate.sql)
        / exp.results_without_covariate.sql
        * 100
    )
    return mase_improvement, sql_improvement


def plot_single_experiment(
    exp: SingleExperimentResult,
    ax_pred: plt.Axes,
    ax_cov: plt.Axes,
    show_legend: bool = True,
    linewidth: float = 2,
    fontsize: float = 12,
    fontweight: str = "semibold",
    use_xlabel: bool = True,
    use_ylabel: bool = True,
):
    """Plots a single experiment's results on the provided axes."""
    x = np.arange(len(exp.data_with_covariate))
    n_train = exp.n_train_timesteps

    pred_no_cov = exp.results_without_covariate.predictions
    quantiles_no_cov = exp.results_without_covariate.quantiles
    pred_with_cov = exp.results_with_covariate.predictions
    quantiles_with_cov = exp.results_with_covariate.quantiles

    ax_pred.plot(
        x,
        exp.data_with_covariate,
        label="Ground Truth",
        color=GROUND_TRUTH_COLOR,
        alpha=0.7,
        linewidth=linewidth,
    )
    ax_pred.plot(
        x[n_train:],
        pred_no_cov,
        label="Pred w/o cov",
        color=PRED_WO_COV_COLOR,
        linewidth=linewidth,
    )
    ax_pred.plot(
        x[n_train:],
        pred_with_cov,
        label="Pred w/ cov",
        color=PRED_W_COV_COLOR,
        linewidth=linewidth,
    )

    test_x = x[n_train:]
    ax_pred.fill_between(
        test_x,
        quantiles_no_cov[0],
        quantiles_no_cov[-1],
        alpha=0.3,
        color=PRED_WO_COV_BAND_COLOR,
    )
    ax_pred.fill_between(
        test_x,
        quantiles_with_cov[0],
        quantiles_with_cov[-1],
        alpha=0.15,
        color=PRED_W_COV_COLOR,
    )

    ax_pred.axvline(
        x=n_train,
        color=TRAIN_TEST_SPLIT_COLOR,
        linestyle="--",
        linewidth=linewidth,
        alpha=0.7,
        label="Train/Test Split",
    )

    if show_legend:
        ax_pred.legend(fontsize=fontsize, loc="upper left")

    ax_pred.grid(False)
    if use_xlabel:
        ax_pred.set_xlabel("Time", fontsize=fontsize, fontweight=fontweight)
    if use_ylabel:
        ax_pred.set_ylabel("Target", fontsize=fontsize, fontweight=fontweight)
    ax_pred.tick_params(labelsize=fontsize, labelbottom=False, bottom=False)

    # Set axes linewidth
    for spine in ax_pred.spines.values():
        spine.set_linewidth(linewidth)

    cmap = plt.cm.get_cmap("Set2")
    for i, cov_signal in enumerate(exp.individual_covariates):
        label = f"{exp.covariate_types[i]} (w: {exp.covariate_weights[i]:.2f})"
        ax_cov.plot(x, cov_signal, linewidth=linewidth, label=label, color=cmap(i))

    ax_cov.axvline(
        x=n_train,
        color=TRAIN_TEST_SPLIT_COLOR,
        linestyle="--",
        linewidth=linewidth,
        alpha=0.9,
        label="Train/Test Split",
    )

    if show_legend:
        ax_cov.legend(fontsize=fontsize, loc="upper left")

    ax_cov.grid(False)
    if use_ylabel:
        ax_cov.set_ylabel("Cov.", fontsize=fontsize, fontweight=fontweight)
    if use_xlabel:
        ax_cov.set_xlabel("Time", fontsize=fontsize, fontweight=fontweight)
    ax_cov.tick_params(labelsize=fontsize)

    # Set axes linewidth
    for spine in ax_cov.spines.values():
        spine.set_linewidth(linewidth)

    for ax in [ax_pred, ax_cov]:
        ax.margins(x=0.01, y=0.2)
        ax.yaxis.set_label_coords(-0.085, 0.5)  # Align y-axis labels


def create_prediction_visualizations(study_results: StudyResult, output_file: Path):
    """Create prediction visualization plots."""

    logger.info(f"Creating prediction visualizations, output file: {output_file}")

    experiments = study_results.exp_results
    n_samples = study_results.n_samples
    covariate_types = study_results.covariate_types

    with PdfPages(output_file) as pdf:
        n_detailed = min(16, n_samples)
        experiments_per_page = 8

        for page_start in range(0, n_detailed, experiments_per_page):
            page_end = min(page_start + experiments_per_page, n_detailed)

            fig = plt.figure(figsize=(16, 24))
            main_gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.25)

            for exp_idx, exp_num in enumerate(range(page_start, page_end)):
                exp_row = exp_idx // 2
                exp_col = exp_idx % 2

                exp_gs = main_gs[exp_row, exp_col].subgridspec(2, 1, hspace=0.05)
                ax_pred = fig.add_subplot(exp_gs[0])
                ax_cov = fig.add_subplot(exp_gs[1], sharex=ax_pred)

                exp = experiments[exp_num]
                plot_single_experiment(exp, ax_pred, ax_cov, exp_num)
                mase_improvement, sql_improvement = calculate_improvement(exp)

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

            plt.suptitle(
                f"Prediction Visualizations: {', '.join(covariate_types).replace('_', ' ').title()} "
                f"(Page {page_start // experiments_per_page + 1})",
                fontsize=24,
                fontweight="bold",
            )
            pdf.savefig(fig)
            plt.close()

    logger.info(f"Prediction visualizations saved to: {output_file}")
