"""Parse Joey-NMT training logs and produce:
  1. A LaTeX table of validation step vs. perplexity for each run.
  2. A line plot of validation perplexity over training steps.

Usage:
    python scripts/analyze_validation.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "analysis"
OUT_DIR.mkdir(exist_ok=True)

RUNS = [
    ("baseline", ROOT / "models" / "baseline.log"),
    ("post-norm", ROOT / "models" / "deen_transformer_postnorm" / "train.log"),
    ("pre-norm", ROOT / "models" / "deen_transformer_prenorm" / "train.log"),
]

# Distinct, colour-blind friendly hues (Okabe-Ito).
COLOURS = {
    "baseline": "#000000",
    "post-norm": "#D55E00",
    "pre-norm": "#0072B2",
}

STEP_RE = re.compile(r"Step:\s*(\d+),")
PPL_RE = re.compile(r"Evaluation result.*?ppl:\s*([0-9]+\.[0-9]+)")


def parse_log(path: Path) -> list[tuple[int, float]]:
    """Return ordered (step, ppl) pairs by pairing the most recent training
    step log line with the subsequent evaluation result line."""
    pairs: list[tuple[int, float]] = []
    last_step: int | None = None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m_step = STEP_RE.search(line)
            if m_step and "Batch Loss" in line:
                last_step = int(m_step.group(1))
                continue
            m_ppl = PPL_RE.search(line)
            if m_ppl and last_step is not None:
                pairs.append((last_step, float(m_ppl.group(1))))
    return pairs


def write_latex_table(data: dict[str, list[tuple[int, float]]], path: Path) -> None:
    steps = sorted({s for run in data.values() for s, _ in run})
    lookup = {name: dict(pairs) for name, pairs in data.items()}
    names = list(data.keys())

    col_spec = "r" + "r" * len(names)
    header = " & ".join(["Step"] + [f"PPL ({n})" for n in names]) + r" \\"

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Validation perplexity per training step for the three NMT runs.}",
        r"  \label{tab:validation-ppl}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    {header}",
        r"    \midrule",
    ]

    for step in steps:
        row_cells = [f"{step}"]
        for name in names:
            val = lookup[name].get(step)
            row_cells.append(f"{val:.2f}" if val is not None else "--")
        lines.append("    " + " & ".join(row_cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_runs(data: dict[str, list[tuple[int, float]]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, pairs in data.items():
        if not pairs:
            continue
        steps, ppls = zip(*pairs)
        ax.plot(
            steps,
            ppls,
            label=name,
            color=COLOURS.get(name, None),
            linewidth=2,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Validation perplexity over training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    data: dict[str, list[tuple[int, float]]] = {}
    for name, log_path in RUNS:
        if not log_path.exists():
            print(f"WARN: missing {log_path}")
            continue
        pairs = parse_log(log_path)
        print(f"{name:>10}: {len(pairs)} validation points (from {log_path.name})")
        data[name] = pairs

    write_latex_table(data, OUT_DIR / "validation_ppl_table.tex")
    plot_runs(data, OUT_DIR / "validation_ppl_plot.png")
    print(f"Wrote LaTeX table to {OUT_DIR / 'validation_ppl_table.tex'}")
    print(f"Wrote plot to {OUT_DIR / 'validation_ppl_plot.png'} (and .pdf)")


if __name__ == "__main__":
    main()
