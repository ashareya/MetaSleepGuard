"""Run a real Sleep-EDF dual-channel baseline with subject-grouped evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from MetaSleepGuard.datasets.public_sleep.loaders import find_sleep_edf_pairs, _sleep_edf_subject_id
from MetaSleepGuard.experiments.common import repo_root
from MetaSleepGuard.models.public_sleep_real_baseline import (
    FIVE_CLASS_LABELS,
    extract_traditional_features,
    grouped_random_forest_baseline,
)


CHANNEL_TARGETS = ("Fpz-Cz", "Pz-Oz")
ANNOTATION_EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}
EVENT_LABEL = {1: "W", 2: "N1", 3: "N2", 4: "N3", 5: "REM"}
CURRENT_PHYSIONET_BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=str(repo_root() / "MetaSleepGuard/data/public_sleep/sleep_edf_raw"),
    )
    parser.add_argument(
        "--output-root",
        default=str(repo_root() / "outputs/metasleepguard_outputs/public_sleep_real_baseline"),
    )
    parser.add_argument("--subjects", type=int, default=5)
    parser.add_argument("--recording", type=int, default=1, choices=[1, 2])
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if args.subjects < 5:
        raise SystemExit("The submission baseline requires at least five subjects")
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    pairs = ensure_sleep_edf_data(data_root, list(range(args.subjects)), args.recording, not args.no_download)
    dataset = load_feature_dataset(pairs[: args.subjects])
    if len(dataset["subject_summaries"]) < 5:
        raise RuntimeError("Fewer than five valid dual-channel subjects remained after channel/annotation checks")
    output_root.mkdir(parents=True, exist_ok=True)
    results = write_baseline_outputs(output_root, dataset)

    from MetaSleepGuard.reports.submission_pack import generate_submission_materials

    generate_submission_materials(repo_root(), section="all")
    print(f"real_sleep_edf_subjects={len(dataset['subject_summaries'])}")
    for subject, count in dataset["subject_epoch_counts"].items():
        print(f"subject_epochs[{subject}]={count}")
    for task in ("3class", "5class"):
        metrics = results[task]
        print(
            f"{task}: accuracy={metrics['accuracy']:.6f} "
            f"macro_f1={metrics['macro_f1']:.6f} kappa={metrics['cohen_kappa']:.6f}"
        )
    print(f"report_html={output_root / 'sleep_edf_real_baseline_report.html'}")
    print(f"summary_json={output_root / 'summary.json'}")


def ensure_sleep_edf_data(data_root: Path, subjects: list[int], recording: int, allow_download: bool) -> list[tuple[Path, Path]]:
    data_root.mkdir(parents=True, exist_ok=True)
    pairs = _pairs_for_requested_subjects(data_root, subjects, recording)
    if len(pairs) >= len(subjects):
        return pairs
    if allow_download:
        try:
            import mne
            from mne.datasets.sleep_physionet.age import fetch_data
        except Exception as exc:
            raise RuntimeError("MNE is required to fetch and read Sleep-EDF") from exc
        print(f"mne_version={mne.__version__}")
        try:
            fetch_data(
                subjects=subjects,
                recording=[recording],
                path=data_root,
                base_url=CURRENT_PHYSIONET_BASE_URL,
                on_missing="raise",
                verbose=True,
            )
        except Exception as exc:
            raise RuntimeError(_manual_download_message(data_root, exc)) from exc
        pairs = _pairs_for_requested_subjects(data_root, subjects, recording)
    if len(pairs) < len(subjects):
        raise RuntimeError(_manual_download_message(data_root))
    return pairs


def _pairs_for_requested_subjects(root: Path, subjects: list[int], recording: int) -> list[tuple[Path, Path]]:
    available = find_sleep_edf_pairs(root)
    selected = []
    for subject in subjects:
        subject_id = f"SC{400 + subject:03d}"
        prefix = f"{subject_id}{recording}"
        match = next((pair for pair in available if pair[0].name.upper().startswith(prefix)), None)
        if match:
            selected.append(match)
    return selected


def _manual_download_message(data_root: Path, cause: Exception | None = None) -> str:
    detail = f" Download error: {cause}" if cause else ""
    return (
        f"Sleep-EDF download is incomplete.{detail} Please place matching Sleep-EDF PSG.edf and "
        f"Hypnogram.edf files in {data_root} and rerun. Real OpenBCI data cannot replace public "
        "expert-labeled sleep data for accuracy evaluation."
    )


def resolve_channels(channel_names: list[str]) -> list[str]:
    selected = []
    for target in CHANNEL_TARGETS:
        normalized_target = _normalize_channel(target)
        matches = [name for name in channel_names if _normalize_channel(name).endswith(normalized_target)]
        if not matches:
            raise ValueError(f"required Sleep-EDF channel is missing: {target}")
        selected.append(matches[0])
    return selected


def load_feature_dataset(pairs: list[tuple[Path, Path]]) -> dict:
    try:
        import mne
    except Exception as exc:
        raise RuntimeError("MNE is required to load Sleep-EDF") from exc
    feature_blocks = []
    labels_all: list[str] = []
    subjects_all: list[str] = []
    epoch_manifest: list[dict] = []
    subject_summaries: list[dict] = []
    feature_names: list[str] | None = None
    skipped: list[dict] = []
    for psg_path, hypnogram_path in pairs:
        subject_id = _sleep_edf_subject_id(psg_path)
        print(f"loading_subject={subject_id} psg={psg_path.name}")
        raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
        try:
            selected_channels = resolve_channels(list(raw.ch_names))
        except ValueError as exc:
            skipped.append({"subject_id": subject_id, "reason": str(exc), "psg_path": str(psg_path)})
            continue
        annotations = mne.read_annotations(hypnogram_path)
        raw.set_annotations(annotations, emit_warning=False)
        crop_start, crop_end = _standard_sleep_crop(raw)
        raw.crop(tmin=crop_start, tmax=crop_end, include_tmax=False)
        raw.pick(selected_channels)
        raw.load_data()
        raw.filter(0.3, 35.0, picks="all", verbose="ERROR")
        events, _ = mne.events_from_annotations(
            raw,
            event_id=ANNOTATION_EVENT_ID,
            chunk_duration=30.0,
            verbose="ERROR",
        )
        if not len(events):
            skipped.append({"subject_id": subject_id, "reason": "no valid 30-second sleep annotations", "psg_path": str(psg_path)})
            continue
        sfreq = float(raw.info["sfreq"])
        epochs = mne.Epochs(
            raw,
            events,
            event_id=ANNOTATION_EVENT_ID,
            tmin=0.0,
            tmax=30.0 - 1.0 / sfreq,
            baseline=None,
            preload=True,
            reject_by_annotation=True,
            verbose="ERROR",
        )
        data = epochs.get_data(copy=False)
        labels = [EVENT_LABEL[int(event_code)] for event_code in epochs.events[:, 2]]
        features, current_names = extract_traditional_features(data, sfreq, CHANNEL_TARGETS)
        if feature_names is None:
            feature_names = current_names
        elif current_names != feature_names:
            raise RuntimeError("feature schema changed between subjects")
        feature_blocks.append(features)
        labels_all.extend(labels)
        subjects_all.extend([subject_id] * len(labels))
        counts = Counter(labels)
        subject_summaries.append(
            {
                "subject_id": subject_id,
                "n_epochs": len(labels),
                "class_distribution": dict(counts),
                "sampling_rate_hz": sfreq,
                "channels": list(CHANNEL_TARGETS),
                "psg_path": str(psg_path),
                "hypnogram_path": str(hypnogram_path),
                "psg_sha1": _sha1(psg_path),
                "hypnogram_sha1": _sha1(hypnogram_path),
            }
        )
        for local_index, (event, label) in enumerate(zip(epochs.events, labels)):
            epoch_manifest.append(
                {
                    "epoch_row": len(epoch_manifest),
                    "subject_id": subject_id,
                    "recording": psg_path.name,
                    "epoch_index_within_subject": local_index,
                    "epoch_start_sec": float((event[0] - raw.first_samp) / sfreq),
                    "label_5class": label,
                    "channels": "Fpz-Cz|Pz-Oz",
                    "sampling_rate_hz": sfreq,
                    "epoch_duration_sec": 30.0,
                    "psg_path": str(psg_path),
                    "hypnogram_path": str(hypnogram_path),
                }
            )
        del data, epochs, raw
    if not feature_blocks:
        raise RuntimeError("No valid Sleep-EDF subjects could be loaded")
    return {
        "features": np.vstack(feature_blocks),
        "feature_names": feature_names or [],
        "labels": np.asarray(labels_all, dtype=str),
        "subject_ids": np.asarray(subjects_all, dtype=str),
        "epoch_manifest": epoch_manifest,
        "subject_summaries": subject_summaries,
        "subject_epoch_counts": {row["subject_id"]: row["n_epochs"] for row in subject_summaries},
        "skipped": skipped,
    }


def _standard_sleep_crop(raw) -> tuple[float, float]:
    scored_sleep = [
        (float(onset), float(duration))
        for onset, duration, description in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)
        if description in ANNOTATION_EVENT_ID and description != "Sleep stage W"
    ]
    if not scored_sleep:
        return 0.0, float(raw.times[-1])
    start = max(0.0, scored_sleep[0][0] - 30.0 * 60.0)
    end = min(float(raw.times[-1]), scored_sleep[-1][0] + scored_sleep[-1][1] + 30.0 * 60.0)
    return start, end


def write_baseline_outputs(output_root: Path, dataset: dict) -> dict[str, dict]:
    _write_csv(output_root / "sleep_edf_epoch_manifest.csv", dataset["epoch_manifest"])
    results: dict[str, dict] = {}
    split_rows = None
    for task in ("3class", "5class"):
        metrics, predictions, current_splits = grouped_random_forest_baseline(
            dataset["features"],
            dataset["labels"],
            dataset["subject_ids"],
            task,
        )
        metrics.update(
            {
                "dataset": "Sleep-EDF Expanded / Sleep Physionet age subset",
                "recording_night": 1,
                "channels": list(CHANNEL_TARGETS),
                "sampling_rate": sorted({row["sampling_rate_hz"] for row in dataset["subject_summaries"]}),
                "epoch_duration_sec": 30.0,
                "preprocessing": {"bandpass_hz": [0.3, 35.0], "wake_crop_minutes": 30},
                "feature_count": len(dataset["feature_names"]),
                "feature_names": dataset["feature_names"],
                "classifier": "RandomForestClassifier(n_estimators=300, class_weight=balanced_subsample)",
                "data_limitations": [
                    "Five-subject small-sample baseline; not a full-cohort benchmark.",
                    "One recording night per subject.",
                    "Pooled out-of-fold metrics from subject-grouped folds.",
                    "No OpenBCI recordings are used for sleep-staging accuracy.",
                ],
            }
        )
        for prediction, epoch in zip(predictions, dataset["epoch_manifest"]):
            prediction.update(
                {
                    "epoch_index_within_subject": epoch["epoch_index_within_subject"],
                    "epoch_start_sec": epoch["epoch_start_sec"],
                    "label_5class": epoch["label_5class"],
                }
            )
        _write_json(output_root / f"sleep_edf_{task}_metrics.json", metrics)
        _write_csv(output_root / f"sleep_edf_{task}_predictions.csv", predictions)
        results[task] = metrics
        split_rows = current_splits
    _write_csv(output_root / "sleep_edf_subject_split.csv", split_rows or [])
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion(results["3class"], figures_dir / "confusion_matrix_3class.png")
    _plot_confusion(results["5class"], figures_dir / "confusion_matrix_5class.png")
    _plot_class_distribution(dataset["labels"], figures_dir / "class_distribution.png")
    _plot_subject_counts(dataset["subject_epoch_counts"], figures_dir / "subject_epoch_counts.png")
    summary = {
        "dataset": "Sleep-EDF Expanded / Sleep Physionet",
        "real_public_dataset": True,
        "synthetic_demo": False,
        "n_subjects": len(dataset["subject_summaries"]),
        "subject_ids": [row["subject_id"] for row in dataset["subject_summaries"]],
        "subject_epoch_counts": dataset["subject_epoch_counts"],
        "n_epochs_total": len(dataset["labels"]),
        "channels": list(CHANNEL_TARGETS),
        "sampling_rates_hz": sorted({row["sampling_rate_hz"] for row in dataset["subject_summaries"]}),
        "epoch_duration_sec": 30.0,
        "split_method": results["5class"]["split_method"],
        "subject_overlap_in_any_fold": False,
        "subjects": dataset["subject_summaries"],
        "skipped_records": dataset["skipped"],
        "metrics": {
            task: {key: results[task][key] for key in ("accuracy", "macro_f1", "weighted_f1", "cohen_kappa")}
            for task in ("3class", "5class")
        },
    }
    _write_json(output_root / "summary.json", summary)
    markdown = _report_markdown(summary, results)
    (output_root / "sleep_edf_real_baseline_report.md").write_text(markdown, encoding="utf-8")
    (output_root / "README.md").write_text(_readme(summary, results), encoding="utf-8")
    (output_root / "DATA_LIMITATIONS.md").write_text(_limitations(), encoding="utf-8")
    (output_root / "sleep_edf_real_baseline_report.html").write_text(_report_html(markdown), encoding="utf-8")
    return results


def _report_markdown(summary: dict, results: dict[str, dict]) -> str:
    subject_rows = "\n".join(
        f"| {subject} | {count} |" for subject, count in summary["subject_epoch_counts"].items()
    )
    metric_rows = "\n".join(
        f"| {task} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['weighted_f1']:.4f} | {metrics['cohen_kappa']:.4f} |"
        for task, metrics in results.items()
    )
    return f"""# Sleep-EDF 真实公开数据双导基线报告

## 数据与方法

- 数据：Sleep-EDF Expanded / Sleep Physionet age subset，真实 PSG 与专家 Hypnogram。
- 被试：{summary['n_subjects']} 名，subjects {', '.join(summary['subject_ids'])}，每人 night 1。
- 双导：Fpz-Cz、Pz-Oz。
- 预处理：0.3–35 Hz，原始 {summary['sampling_rates_hz']} Hz，30 秒 Epoch；丢弃 Movement/Unknown。
- 划分：{summary['split_method']}；同一被试不会同时出现在训练和测试集合。
- 模型：RandomForest，传统时域/Hjorth/频带功率/相对功率/比值/谱熵特征。

## 每名被试 Epoch 数

| subject_id | valid_epochs |
|---|---:|
{subject_rows}

## 真实基线指标

| task | Accuracy | Macro-F1 | Weighted-F1 | Cohen's Kappa |
|---|---:|---:|---:|---:|
{metric_rows}

## 图表

![3-class confusion matrix](figures/confusion_matrix_3class.png)

![5-class confusion matrix](figures/confusion_matrix_5class.png)

![Class distribution](figures/class_distribution.png)

![Subject epoch counts](figures/subject_epoch_counts.png)

## 结论边界

这些指标来自真实 Sleep-EDF 专家标签，可作为小样本公开数据基线证据。仅包含 {summary['n_subjects']} 名被试和每人一晚，不代表完整数据集性能。真实 OpenBCI 数据未用于准确率计算，仍只用于采集链路与质量守护验证。
"""


def _readme(summary: dict, results: dict[str, dict]) -> str:
    return f"""# Sleep-EDF Real Baseline

本目录是可提交的真实公开睡眠数据小样本基线，不是 synthetic smoke。

- 被试数：{summary['n_subjects']}
- 总有效 Epoch：{summary['n_epochs_total']}
- 双导：Fpz-Cz / Pz-Oz
- 划分：{summary['split_method']}
- 3-class：Accuracy {results['3class']['accuracy']:.4f}，Macro-F1 {results['3class']['macro_f1']:.4f}，Kappa {results['3class']['cohen_kappa']:.4f}
- 5-class：Accuracy {results['5class']['accuracy']:.4f}，Macro-F1 {results['5class']['macro_f1']:.4f}，Kappa {results['5class']['cohen_kappa']:.4f}

复现命令：

```powershell
.\\run.ps1 -Task public-sleep-real-baseline -Python $py
```
"""


def _limitations() -> str:
    return """# Data Limitations

- 这是至少 5 名被试、每人一晚的真实 Sleep-EDF 小样本基线，不是完整队列结论。
- 指标采用 GroupKFold 被试级划分和 pooled out-of-fold 预测，不存在同一被试跨训练/测试集合。
- 标准睡眠区间前后各保留最多 30 分钟清醒期，避免整日清醒记录主导类别分布。
- 模型是传统 RandomForest 基线；低指标照实报告，没有挑选 Epoch 或被试美化结果。
- OpenBCI 真实数据没有专家睡眠标签，未参与这些准确率指标。
"""


def _report_html(markdown: str) -> str:
    lines = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("!["):
            alt, src = line[2:].split("](", 1)
            lines.append(f'<figure><img src="{src[:-1]}" alt="{alt}"></figure>')
        elif line.startswith("- "):
            lines.append(f"<p>• {line[2:]}</p>")
        elif line.startswith("|"):
            lines.append(f"<pre>{line}</pre>")
        elif line and not line.startswith("```"):
            lines.append(f"<p>{line}</p>")
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Sleep-EDF 真实公开数据双导基线报告</title><style>
body{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1200px;margin:28px;color:#1f2933;line-height:1.55}
h1,h2{color:#16324f}img{max-width:100%;border:1px solid #ccd5df}pre{margin:0;font-family:Consolas,monospace}
</style></head><body>""" + "\n".join(lines) + "</body></html>"


def _plot_confusion(metrics: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    classes = metrics["classes"]
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks(range(len(classes)), classes, rotation=30)
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"Sleep-EDF real baseline: {metrics['task']}")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_class_distribution(labels: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    counts = Counter(map(str, labels))
    values = [counts.get(label, 0) for label in FIVE_CLASS_LABELS]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(FIVE_CLASS_LABELS, values, color=["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"])
    axis.set_ylabel("30-second epochs")
    axis.set_title("Sleep-EDF real baseline class distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_subject_counts(counts: dict[str, int], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(list(counts), list(counts.values()), color="#4c78a8")
    axis.set_ylabel("Valid 30-second epochs")
    axis.set_title("Epoch count by held-out subject")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_channel(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


if __name__ == "__main__":
    main()
