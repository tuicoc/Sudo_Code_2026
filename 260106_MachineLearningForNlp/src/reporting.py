"""Putting the finished experiments side by side: one table, one chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import Config


class ResultsReporter:
    """Renders the comparison of every experiment's saved metrics."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.metrics: list[str] = config.require("report.metrics")
        self.experiments: list[dict] = config.require("experiments")

    def table(self, results: dict[str, dict]) -> pd.DataFrame:
        """One column per experiment, `mean ± std` per metric."""
        return pd.DataFrame(
            {
                label: [f"{r['mean'][m]:.4f} ± {r['std'][m]:.4f}" for m in self.metrics]
                for label, r in results.items()
            },
            index=self.metrics,
        )

    def bar_chart(self, results: dict[str, dict], out_path: Path | None = None) -> Path:
        """Grouped bars per metric, with the fold spread as error bars.

        The error bars are the point: two bars whose intervals overlap have not been shown
        to differ.
        """
        colors = {e["label"]: e["color"] for e in self.experiments}
        figure_config = self.config.require("report.figure")

        x = np.arange(len(self.metrics))
        bar_width = 0.8 / len(results)

        fig, ax = plt.subplots(figsize=(figure_config["width"], figure_config["height"]))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        for i, (label, result) in enumerate(results.items()):
            offset = (i - (len(results) - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                [result["mean"][m] for m in self.metrics],
                width=bar_width * 0.9,
                yerr=[result["std"][m] for m in self.metrics],
                capsize=3,
                color=colors.get(label),
                label=label,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(self.config.require("report.metric_labels"))
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(f"Score ({results[next(iter(results))]['n_splits']}-fold CV mean ± std)")
        ax.set_title(self.config.require("report.title"))
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()

        out_path = out_path or self.config.path("paths.comparison_figure")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=figure_config["dpi"])
        plt.close(fig)
        return out_path
