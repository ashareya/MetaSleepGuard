"""Quality, confidence, and rejection trend plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def plot_quality_trend(grades: Sequence[str], output_path: str | Path, epoch_sec: float = 30.0) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grade_value = {"A": 4, "B": 3, "C": 2, "D": 1}
    y = [grade_value.get(grade, 0) for grade in grades]
    x = [i * epoch_sec / 60.0 for i in range(len(grades))]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
    ax.step(x, y, where="post", linewidth=1.4, color="#2f6f8f")
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["D", "C", "B", "A"])
    ax.set_xlabel("Time (min)")
    ax.set_title("Signal quality trend")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_confidence_trend(confidence: Sequence[float], accepted: Sequence[bool], output_path: str | Path, epoch_sec: float = 30.0) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [i * epoch_sec / 60.0 for i in range(len(confidence))]
    colors = ["#2f6f8f" if ok else "#b33a3a" for ok in accepted]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
    ax.plot(x, confidence, color="#444444", linewidth=1.0)
    ax.scatter(x, confidence, c=colors, s=24)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence and rejection trend")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

