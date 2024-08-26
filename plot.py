# Simple helper file to plot generated auc files.
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import difflib
import jsonlines
from ioh import get_problem, logger
from misc import aoc_logger, correct_aoc, OverBudgetException

try:
    from colorama import Fore, Back, Style, init
    init()
except ImportError:  # fallback so that the imported classes always exist
    class ColorFallback():
        __getattr__ = lambda self, name: ''
    Fore = Back = Style = ColorFallback()

def color_diff(diff):
    for line in diff:
        if line.startswith('+'):
            yield Fore.GREEN + line + Fore.RESET
        elif line.startswith('-'):
            yield Fore.RED + line + Fore.RESET
        elif line.startswith('^'):
            yield Fore.BLUE + line + Fore.RESET
        else:
            yield line

def code_compare(code1, code2, printdiff=False):
    # Parse the Python code into ASTs
    # Use difflib to find differences
    diff = difflib.ndiff(code1.splitlines(), code2.splitlines())
    
    # Count the number of differing lines
    diffs = sum(1 for x in diff if x.startswith("- ") or x.startswith("+ "))
    if printdiff and code1 != "":
        diff = difflib.ndiff(code1.splitlines(), code2.splitlines())
        print('\n'.join(color_diff(diff)))
    # Calculate total lines for the ratio
    total_lines = max(len(code1.splitlines()), len(code2.splitlines()))
    similarity_ratio = (total_lines - diffs) / total_lines if total_lines else 1
    if printdiff:
        print(similarity_ratio)
    return 1 - similarity_ratio


experiments_dirs = [
    "exp-08-20_122254-gpt-4o-2024-05-13-ES gpt-4o-HPO",  # /log.jsonl
    "exp-08-20_123922-gpt-4o-2024-05-13-ES gpt-4o-HPO",
    "exp-08-26_090633-gpt-4o-2024-05-13-ES gpt-4o-HPO",
    "exp-08-26_090643-gpt-4o-2024-05-13-ES gpt-4o-HPO",
    "exp-08-26_095046-gpt-4o-2024-05-13-ES gpt-4o-HPO",
]
budget = 100

label_main = "GPT-4o-HPO"

convergence_lines = []
convergence_default_lines = []
code_diff_ratios_lines = []


for i in range(len(experiments_dirs)):
    convergence = np.zeros(budget)
    convergence_default = np.zeros(budget)
    code_diff_ratios = np.zeros(budget)
    best_so_far = -np.Inf
    best_so_far_default = 0
    previous_code = ""
    previous_name = ""
    log_file = experiments_dirs[i] + "/log.jsonl"
    if os.path.exists(log_file):
        with jsonlines.open(log_file) as reader:
            for obj in reader.iter(type=dict, skip_invalid=True):
                gen = 0
                fitness = None
                code_diff = 0
                code = ""
                if "_solution" in obj.keys():
                    code = obj["_solution"]
                if "_generation" in obj.keys():
                    gen = obj["_generation"]
                if "_fitness" in obj.keys():
                    fitness = obj["_fitness"]
                else:
                    fitness = None

                if fitness <= best_so_far:
                    code_diff = code_compare(previous_code, code, False)
                else:
                    name = obj["_name"]
                    print(f"-- {gen} -- {previous_name} --> {name}")
                    code_diff = code_compare(previous_code, code, True)
                    best_so_far = fitness

                    #check optimized fitness against plain fitness
                    if True:
                        l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
                        aucs = []
                        exec(code, globals())
                        for fid in np.arange(1, 25):
                            for iid in [1, 2, 3]:  # , 4, 5]
                                problem = get_problem(fid, iid, 5)
                                problem.attach_logger(l2)
                                for rep in range(3):
                                    np.random.seed(rep)
                                    try:
                                        #default params
                                        algorithm = globals()[name](budget=10000, dim=5)
                                        algorithm(problem)
                                    except OverBudgetException:
                                        pass
                                    auc = correct_aoc(problem, l2, 10000)
                                    aucs.append(auc)
                                    l2.reset(problem)
                                    problem.reset()

                        best_so_far_default = np.mean(aucs)
                        print("best_so_far_default", best_so_far_default)
                        print("best_so_far", best_so_far)

                    previous_code = code
                    previous_name = name
                
                code_diff_ratios[gen] = code_diff
                convergence[gen] = fitness
                convergence_default[gen] = best_so_far_default

    # now fix the holes
    best_so_far = 0
    best_so_far_d = 0
    for i in range(len(convergence)):
        if convergence[i] >= best_so_far:
            best_so_far = convergence[i]
            best_so_far_d = convergence_default[i]
        else:
            convergence[i] = best_so_far
            convergence_default[i] = best_so_far_d
    convergence_lines.append(convergence)
    convergence_default_lines.append(convergence_default)
    code_diff_ratios_lines.append(code_diff_ratios)


plt.figure(figsize=(6, 4))
for i in range(len(convergence_lines)):
    plt.plot(np.arange(budget), convergence_lines[i], linestyle="dashed")
    plt.plot(np.arange(budget), convergence_default_lines[i], linestyle="dotted")

# convergence curves
mean_convergence = np.array(convergence_lines).mean(axis=0)
mean_convergence_default = np.array(convergence_default_lines).mean(axis=0)
std = np.array(convergence_lines).std(axis=0)
plt.plot(
    np.arange(budget),
    mean_convergence,
    color="b",
    linestyle="solid",
    label=label_main,
)
plt.plot(
    np.arange(budget),
    mean_convergence_default,
    color="r",
    linestyle="solid",
    label=label_main + " default",
)
plt.fill_between(
    np.arange(budget),
    mean_convergence - std,
    mean_convergence + std,
    color="b",
    alpha=0.05,
)
# plt.fill_between(x, 0, 1, where=error_bars, color='r', alpha=0.2)
plt.ylim(0.0, 0.7)
plt.xlim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig(f"plot_aucs_HPO.png")
plt.clf()


# Code diff curves
plt.figure(figsize=(6, 4))
for i in range(len(code_diff_ratios_lines)):
    plt.plot(np.arange(budget), code_diff_ratios_lines[i], linestyle="dashed")

mean_code_diff = np.array(code_diff_ratios_lines).mean(axis=0)
std = np.array(code_diff_ratios_lines).std(axis=0)
plt.plot(
    np.arange(budget),
    mean_code_diff,
    color="b",
    linestyle="solid",
    label=label_main,
)
plt.fill_between(
    np.arange(budget),
    mean_code_diff - std,
    mean_code_diff + std,
    color="b",
    alpha=0.05,
)
# plt.fill_between(x, 0, 1, where=error_bars, color='r', alpha=0.2)
plt.ylim(0.0, 1.0)
plt.xlim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig(f"plot_diffratio_HPO.png")
plt.clf()
