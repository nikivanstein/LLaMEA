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



def check_alg(code, algorithm_name):
    exec(code, globals())
    dim = 5
    budget = 100
    error = ""
    algorithm = None
    # perform a small run to check for any code errors
    l2_temp = aoc_logger(100, upper=1e2, triggers=[logger.trigger.ALWAYS])
    problem = get_problem(11, 1, dim)
    problem.attach_logger(l2_temp)
    try:
        algorithm = globals()[algorithm_name](budget=100, dim=dim)
        algorithm(problem)
    except OverBudgetException:
        pass
    except Exception:
        return False
    return True

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


experiments_dirs= {"10":[
    "exp-09-12_162022-gpt-4o-ES elitism",  # /log.jsonl
    "exp-09-12_190222-gpt-4o-ES elitism",
    "exp-09-12_203704-gpt-4o-ES elitism",
    "exp-09-12_220647-gpt-4o-ES elitism",
    "exp-09-12_225550-gpt-4o-ES elitism",
],
"20": [
    "exp-09-13_112056-gpt-4o-ES elitism",  # /log.jsonl
    "exp-09-12_190222-gpt-4o-ES elitism",
    "exp-09-13_163932-gpt-4o-ES elitism",
    "exp-09-13_194721-gpt-4o-ES elitism",
    "exp-09-13_205228-gpt-4o-ES elitism",
]
, "30": [
    "exp-09-17_113803-gpt-4o-ES elitism",  # /log.jsonl
    "exp-09-17_125603-gpt-4o-ES elitism",
    "exp-09-17_155812-gpt-4o-ES elitism",
    "exp-09-17_191133-gpt-4o-ES elitism",
    "exp-09-17_211812-gpt-4o-ES elitism",
],
"40": [   
    "exp-09-18_141137-gpt-4o-ES elitism",  # /log.jsonl
    "exp-09-18_171028-gpt-4o-ES elitism",
    "exp-09-18_182204-gpt-4o-ES elitism",
    "exp-09-18_211522-gpt-4o-ES elitism",
    "exp-09-18_212621-gpt-4o-ES elitism",
    ]
}
budget = 100

label_main = "LLaMEA"

convergence_lines = []
convergence_default_lines = []
code_diff_ratios_lines = []


best_code = ""
best_config = ""

for x in ["10","20","30","40"]:
    exp_dirs = experiments_dirs[x]

    for i in range(len(exp_dirs)):
        convergence = np.zeros(budget)
        convergence_default = np.zeros(budget)
        code_diff_ratios = np.zeros(budget)
        best_so_far = -np.Inf
        best_so_far_default = 0
        previous_code = ""
        previous_name = ""
        log_file = f"runs/x_mutation/{x}%/" + exp_dirs[i] + "/conversationlog.jsonl"
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
                        code_diff = code_compare(previous_code, code, False)
                        best_so_far = fitness
                        best_code = code
                        best_config = obj["incumbent"]

                        #check optimized fitness against plain fitness
                        if True:
                            l2 = aoc_logger(10000, upper=1e2, triggers=[logger.trigger.ALWAYS])
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
            print("best code", best_code)
            print("best config", best_config)

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


    plt.figure(figsize=(6, 6))
    for i in range(len(convergence_lines)):
        plt.plot(np.arange(budget), convergence_lines[i], linestyle="dashed", color="C0")
        plt.plot(np.arange(budget), convergence_default_lines[i], linestyle="dotted", color="C4")

    # convergence curves

    np.save("HPOconvergence_lines.npy", convergence_lines)
    mean_convergence = np.array(convergence_lines).mean(axis=0)
    mean_convergence_default = np.array(convergence_default_lines).mean(axis=0)
    std = np.array(convergence_lines).std(axis=0)
    std_d = np.array(convergence_default_lines).std(axis=0)
    plt.plot(
        np.arange(budget),
        mean_convergence,
        color="C0",
        linestyle="solid",
        label=label_main,
    )
    plt.plot(
        np.arange(budget),
        mean_convergence_default,
        color="C4",
        linestyle="solid",
        label=label_main + " default",
    )
    plt.fill_between(
        np.arange(budget),
        mean_convergence_default - std_d,
        mean_convergence_default + std_d,
        color="C4",
        alpha=0.05,
    )
    plt.fill_between(
        np.arange(budget),
        mean_convergence - std,
        mean_convergence + std,
        color="C0",
        alpha=0.05,
    )
    # plt.fill_between(x, 0, 1, where=error_bars, color='r', alpha=0.2)
    plt.ylim(0.0, 0.7)
    plt.xlim(0, 100)
    plt.xlabel("LLM iterations")
    plt.ylabel("Mean AOCC")
    plt.title("optimized vs non-optimized performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_aucs_HPO.png")


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
