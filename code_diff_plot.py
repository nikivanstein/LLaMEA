import os
import json
import difflib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


budegt = 100


def code_compare(code1, code2):
    modified_lines = 0
    total_lines = max(len(code1), len(code2))
    matcher = difflib.SequenceMatcher(None, code1, code2)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            modified_lines += max(i2 - i1, j2 - j1)
        elif tag in ('delete', 'insert'):
            modified_lines += (i2 - i1) if tag == 'delete' else (j2 - j1)
    diff_ratio = modified_lines / total_lines if total_lines > 0 else 0
    return diff_ratio


def loads_code_files(exp_dir):
    codes = {}
    code_parents = {}
    log_file = f"{exp_dir}/log.jsonl"
    log_data = []
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            log_data += [data]
    for i in range(len(log_data)):
        code_id = log_data[i]["id"]
        code_content = log_data[i]["solution"]
        code_parent = log_data[i]["parent_id"]
        codes[code_id] = code_content
        code_parents[code_id] = code_parent
    return codes, code_parents


def calculate_code_diff(codes, code_parents):
    code_diffs = {}
    for code_id in codes:
        code = codes[code_id]
        parent_id = code_parents[code_id]
        if parent_id is None:
            code_diffs[code_id] = 0
            continue
        parent_code = codes[parent_id]
        code_diff = code_compare(parent_code, code)
        code_diffs[code_id] = code_diff
    return code_diffs


def box_plot_mutation_code_diff(exp_dirs, labels, title):
    data = []
    for i in range(len(exp_dirs)):
        mutation_exps = exp_dirs[i]
        mutation_label = labels[i]
        # mutation_diff_data = []
        for exp_dir in mutation_exps:
            codes, code_parents = loads_code_files(exp_dir)
            code_diff = calculate_code_diff(codes, code_parents)
            code_diff_values = list(code_diff.values())
            for code_diff_value in code_diff_values:
                if code_diff_value <= 0 or code_diff_value > 0.8:
                    continue
                data += [[int(mutation_label), code_diff_value]]
    df = pd.DataFrame(data, columns=["Mutation", "Code Difference"])
    for mutation_label in labels:
        if int(mutation_label) not in df["Mutation"].values:
            df = pd.concat([df, pd.DataFrame(
                [[int(mutation_label), 0]], columns=["Mutation", "Code Difference"])])
    sns.violinplot(x="Mutation", y="Code Difference", data=df, inner="box",
                   palette="muted", hue="Mutation", legend=False)
    plt.xticks(rotation=45)
    plt.xlabel("Mutation")
    plt.ylabel("Code Difference")
    plt.title("Code Difference for Different Mutations")
    plt.savefig(title)
    plt.cla()


if __name__ == "__main__":
    # models = ["test", "test2", "test3-gpt-3.5-turbo", "test3-gpt-4o",
    #           "gpt-4o", "gpt-3.5-turbo",
    #           "Llama-3.2-1B", "Llama-3.2-3B",
    #           "Meta-Llama-3.1-8B", "Meta-Llama-3.1-70B"]
    models = [f"prompt{i}-{m}" for i in range(1, 6)
              for m in ["gpt-3.5-turbo", "gpt-4o"]]
    mutations = ["beta1.5", "2"] + [str(i*5) for i in range(1, 11)]
    for model in models:
        exp_dirs = []
        labels = []
        for mutation in mutations:
            if not os.path.exists(f"exp-{model}/{mutation}"):
                continue
            if "test" in model and mutation == "30":
                continue
            folders = os.listdir(
                f"exp-{model}/{mutation}")
            exp_dirs += [
                [f"exp-{model}/{mutation}/{f}" for f in folders if f.startswith("exp")]]
            labels += [f"{mutation}"]
        title = f"results/code_diff/{model}-code-diff.png"
        box_plot_mutation_code_diff(exp_dirs, labels, title)
