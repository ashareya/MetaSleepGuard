"""Sleep-stage trend plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def plot_hypnogram(stages: Sequence[str], output_path: str | Path, epoch_sec: float = 30.0) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = {"W": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0, "LIGHT": 1.5, "NREM": 1, "暂不判定": -1}
    y = [order.get(stage, -1) for stage in stages]
    x = [i * epoch_sec / 60.0 for i in range(len(stages))]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
    ax.step(x, y, where="post", linewidth=1.4)
    ax.set_yticks([-1, 0, 1, 2, 3, 4])
    ax.set_yticklabels(["Reject", "N3", "N2/NREM", "N1", "REM", "W"])
    ax.set_xlabel("Time (min)")
    ax.set_title("Sleep-stage trend")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

