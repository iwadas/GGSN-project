"""Pareto frontier utilities for architecture search results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def compute_pareto_frontier(
    results: pd.DataFrame,
    accuracy_column: str = "validation_accuracy",
    parameter_column: str = "parameters",
    latency_column: str = "latency_ms",
) -> pd.DataFrame:
    """Return rows not dominated on accuracy, parameter count, and latency."""
    required_columns = {accuracy_column, parameter_column, latency_column}
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing Pareto columns: {missing}.")

    candidates = results.dropna(
        subset=[accuracy_column, parameter_column, latency_column]
    ).copy()
    if candidates.empty:
        return candidates

    is_pareto = []
    for _, row in candidates.iterrows():
        dominates_row = (
            (candidates[accuracy_column] >= row[accuracy_column])
            & (candidates[parameter_column] <= row[parameter_column])
            & (candidates[latency_column] <= row[latency_column])
            & (
                (candidates[accuracy_column] > row[accuracy_column])
                | (candidates[parameter_column] < row[parameter_column])
                | (candidates[latency_column] < row[latency_column])
            )
        )
        is_pareto.append(not bool(dominates_row.any()))

    pareto = candidates.loc[is_pareto].copy()
    return pareto.sort_values(
        by=[accuracy_column, latency_column, parameter_column],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def save_pareto_frontier(
    results_csv_path: str | Path,
    output_csv_path: str | Path,
    accuracy_column: str = "validation_accuracy",
    parameter_column: str = "parameters",
    latency_column: str = "latency_ms",
) -> pd.DataFrame:
    """Compute and save Pareto-efficient rows from a search history CSV."""
    results = pd.read_csv(results_csv_path)
    pareto = compute_pareto_frontier(
        results=results,
        accuracy_column=accuracy_column,
        parameter_column=parameter_column,
        latency_column=latency_column,
    )

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pareto.to_csv(output_csv_path, index=False)
    return pareto


def plot_accuracy_vs_latency(
    results: pd.DataFrame,
    pareto: pd.DataFrame,
    output_path: str | Path,
    accuracy_column: str = "validation_accuracy",
    latency_column: str = "latency_ms",
) -> None:
    """Plot accuracy against inference latency and highlight Pareto models."""
    _plot_tradeoff(
        results=results,
        pareto=pareto,
        x_column=latency_column,
        y_column=accuracy_column,
        x_label="Latency (ms)",
        y_label="Validation accuracy",
        title="Accuracy vs Latency",
        output_path=output_path,
    )


def plot_accuracy_vs_parameters(
    results: pd.DataFrame,
    pareto: pd.DataFrame,
    output_path: str | Path,
    accuracy_column: str = "validation_accuracy",
    parameter_column: str = "parameters",
) -> None:
    """Plot accuracy against parameter count and highlight Pareto models."""
    _plot_tradeoff(
        results=results,
        pareto=pareto,
        x_column=parameter_column,
        y_column=accuracy_column,
        x_label="Parameters",
        y_label="Validation accuracy",
        title="Accuracy vs Parameters",
        output_path=output_path,
    )


def plot_pareto_frontier(
    results: pd.DataFrame,
    pareto: pd.DataFrame,
    output_path: str | Path,
    accuracy_column: str = "validation_accuracy",
    parameter_column: str = "parameters",
    latency_column: str = "latency_ms",
) -> None:
    """Plot the Pareto frontier in latency-parameter space."""
    _plot_tradeoff(
        results=results,
        pareto=pareto,
        x_column=latency_column,
        y_column=parameter_column,
        x_label="Latency (ms)",
        y_label="Parameters",
        title="Pareto Frontier",
        output_path=output_path,
        color_column=accuracy_column,
    )


def plot_pareto_analysis(
    results_csv_path: str | Path,
    pareto_csv_path: str | Path,
    accuracy_latency_path: str | Path,
    accuracy_parameters_path: str | Path,
    pareto_frontier_path: str | Path,
) -> pd.DataFrame:
    """Save Pareto CSV and all Stage 8 trade-off plots."""
    results = pd.read_csv(results_csv_path)
    pareto = compute_pareto_frontier(results)

    pareto_csv_path = Path(pareto_csv_path)
    pareto_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pareto.to_csv(pareto_csv_path, index=False)

    plot_accuracy_vs_latency(results, pareto, accuracy_latency_path)
    plot_accuracy_vs_parameters(results, pareto, accuracy_parameters_path)
    plot_pareto_frontier(results, pareto, pareto_frontier_path)

    return pareto


def _plot_tradeoff(
    results: pd.DataFrame,
    pareto: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: str | Path,
    color_column: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    plot_results = results.dropna(subset=[x_column, y_column]).copy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    if plot_results.empty:
        ax.set_title(f"{title} (no latency data)")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    if color_column is not None and color_column in plot_results.columns:
        scatter = ax.scatter(
            plot_results[x_column],
            plot_results[y_column],
            c=plot_results[color_column],
            alpha=0.45,
            label="all candidates",
        )
        fig.colorbar(scatter, ax=ax, label=color_column)
    else:
        ax.scatter(
            plot_results[x_column],
            plot_results[y_column],
            alpha=0.45,
            label="all candidates",
        )

    if not pareto.empty:
        ax.scatter(
            pareto[x_column],
            pareto[y_column],
            marker="x",
            s=80,
            linewidths=2,
            label="pareto efficient",
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
