# Simple helper file to plot generated auc files.
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import tqdm
import ast
import difflib
import jellyfish
import jsonlines


def code_compare(code1, code2, printdiff=False):
    # Parse the Python code into ASTs
    # Use difflib to find differences
    diff = difflib.ndiff(code1.splitlines(), code2.splitlines())
    # Count the number of differing lines
    diffs = sum(1 for x in diff if x.startswith("- ") or x.startswith("+ "))
    # Calculate total lines for the ratio
    total_lines = max(len(code1.splitlines()), len(code2.splitlines()))
    similarity_ratio = (total_lines - diffs) / total_lines if total_lines else 1
    return 1 - similarity_ratio


experiments = [["exp-08-20_103957-gpt-4o-2024-05-13-ES gpt-4o-HPO/log.jsonl"]]
budget = 100

colors = ["b"]
linestyles = ["solid"]
labels = ["GPT-4o-HPO"]

for i in range(len(experiments)):
    exp_logs = experiments[i]
    color = colors[i]
    ls = linestyles[i]
    label = labels[i]
    mean_aucs = []
    current_best = 0
    best_aucs = []
    std_aucs = []
    error_bars = []
    best_try = ""

    m_aucs = []
    log_i = 1
    for log_file in exp_logs:
        if os.path.exists(log_file):
            with jsonlines.open(log_file) as reader:
                for obj in reader.iter(type=dict, skip_invalid=True):
                    if "aucs" in obj.keys():
                        aucs = np.array(obj["aucs"])
                    else:
                        aucs = []
                    if np.mean(aucs) > current_best:
                        current_best = np.mean(aucs)
                        best_try = str(obj["_generation"]) + obj["_name"]
                    if len(aucs) > 0:
                        mean_aucs.append(np.mean(aucs))
                        std_aucs.append(np.std(aucs))
                    else:
                        mean_aucs.append(np.nan)
                        std_aucs.append(np.nan)
                    best_aucs.append(current_best)

            mean_aucs = np.array(mean_aucs)
            std_aucs = np.array(std_aucs)
            error_bars = np.array(error_bars)
            x = np.arange(budget)

            plt.plot(
                np.arange(len(mean_aucs)),
                mean_aucs,
                color=color,
                linestyle=ls,
                label=f"{label} {log_i}",
            )
            plt.fill_between(
                np.arange(len(mean_aucs)),
                mean_aucs - std_aucs,
                mean_aucs + std_aucs,
                color=color,
                alpha=0.05,
            )
            # plt.fill_between(x, 0, 1, where=error_bars, color='r', alpha=0.2)
            plt.ylim(0.0, 0.7)
            plt.xlim(0, 100)
            log_i += 1

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_aucs_{labels[i]}.png")
    plt.clf()
