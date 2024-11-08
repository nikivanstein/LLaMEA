import os
import json
import difflib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


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


def build_data(exp_dirs, prompt, llm, labels):
    data = []
    for i in range(len(exp_dirs)):
        mutation_exps = exp_dirs[i]
        mutation_label = labels[i]
        for exp_dir in mutation_exps:
            codes, code_parents = loads_code_files(exp_dir)
            code_diff = calculate_code_diff(codes, code_parents)
            code_diff_values = list(code_diff.values())
            for code_diff_value in code_diff_values:
                if code_diff_value <= 0 or code_diff_value > 0.8:
                    continue
                data += [[llm, prompt, int(mutation_label), code_diff_value]]
    return data


def violin_plot(df, prompt, llm, title):
    sns.violinplot(x="Mutation", y="Code Difference", data=df, cut=0,
                   inner="stick", palette="muted", hue="Mutation",
                   legend=False)
    plt.xlabel("requested mutation rate")
    plt.ylabel("code difference")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.xticks(ticks=list(range(5)), labels=["2%", "5%", "10%", "20%", "40%"])
    plt.title(f"Code Difference Distribution\nwhen Using {prompt} with {llm}")
    plt.tight_layout()
    plt.savefig(title)
    plt.cla()


def ratio_plot(df, prompt, llm, title):
    df_temp = df.copy()
    df_temp["ratio"] = (df["Code Difference"] * 100) / df["Mutation"]
    sns.stripplot(x="Mutation", y="ratio", data=df_temp, jitter=True,
                  palette="muted", hue="Mutation", legend=False)
    plt.axhline(y=1, color='r', linestyle='--',
                label="delivered code difference = requested mutation rate")
    # plt.yscale("log")
    plt.xticks(ticks=list(range(5)), labels=["2%", "5%", "10%", "20%", "40%"])
    plt.xlabel("requested mutation rate")
    plt.ylabel("ratio")
    plt.title(
        f"Ratio of Delivered Code Difference to Requested Mutation Rate\nwhen Using {prompt} with {llm}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(title)
    plt.cla()


def MSE(df, llm, prompt):
    MSE = [llm, prompt]
    for mutation in [2, 5, 10, 20, 40]:
        df_temp = df[df["Mutation"] == mutation]
        if df_temp.values.all() == 0:
            MSE += [np.nan]
        else:
            MSE += [((df_temp["Code Difference"]*100/mutation -
                    df_temp["Mutation"]/mutation)**2).mean()]
    return MSE


def save_MSE_table(df):
    MSE_table = []
    for llm in ["gpt-3.5-turbo", "gpt-4o"]:
        for prompt in [f"prompt{i}" for i in range(1, 12)]:
            df_temp = df[(df["model"] == llm) & (df["prompt"] == prompt)]
            if df_temp.empty:
                continue
            MSE_table += [MSE(df_temp)]
    MSE_table = pd.DataFrame(
        MSE_table, columns=["model", "prompt", "2%", "5%", "10", "20", "40"])
    MSE_table.to_csv("results/code_diff/MSE_table.csv")


def violin_mutation_plot(df, llm, mutation, title):
    sns.violinplot(x="prompt", y="Code Difference", data=df, cut=0,
                   inner="stick", palette="muted", hue="prompt",
                   legend=False)
    plt.axhline(y=mutation/100, color='r', linestyle='--',
                label="requested mutation rate")
    plt.xlabel("different prompts")
    plt.ylabel("code difference")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.title(
        f"Code Difference Distribution of Different Prompts\nwhen Requesting {mutation}% Mutation Rate with {llm}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(title)
    plt.cla()


def ratio_mutation_plot(df, llm, mutation, title):
    df_temp = df.copy()
    df_temp["ratio"] = (df["Code Difference"] * 100) / mutation
    sns.stripplot(x="prompt", y="ratio", data=df_temp, jitter=True,
                  palette="muted", hue="prompt", legend=False)
    plt.axhline(y=1, color='r', linestyle='--',
                label="delivered code difference = requested mutation rate")
    # plt.yscale("log")
    plt.xlabel("different prompts")
    plt.ylabel("ratio")
    plt.title(
        f"Ratio of Delivered Code Difference to Requested Mutation Rate\nwhen Requesting {mutation}% Mutation Rate with {llm}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(title)
    plt.cla()


if __name__ == "__main__":
    # models = ["test", "test2", "test3-gpt-3.5-turbo", "test3-gpt-4o",
    #           "gpt-4o", "gpt-3.5-turbo",
    #           "Llama-3.2-1B", "Llama-3.2-3B",
    #           "Meta-Llama-3.1-8B", "Meta-Llama-3.1-70B"]
    models = [f"prompt{i}_{m}" for m in ["gpt-3.5-turbo", "gpt-4o"]
              for i in range(1, 12)]
    mutations = ["beta1.5", "2"] + [str(i*5) for i in range(1, 11)]
    df_values = []
    for model in models:
        prompt = model.split("_")[0]
        llm = model.split("_")[1]
        exp_name = f"{prompt}-{llm}"
        exp_dirs = []
        labels = []
        if not os.path.exists(f"exp-{exp_name}"):
            continue
        for mutation in mutations:
            if not os.path.exists(f"exp-{exp_name}/{mutation}"):
                continue
            folders = os.listdir(
                f"exp-{exp_name}/{mutation}")
            exp_dirs += [
                [f"exp-{exp_name}/{mutation}/{f}" for f in folders if f.startswith("exp")]]
            labels += [f"{mutation}"]
        df_values += build_data(exp_dirs, prompt, llm, labels)
    df = pd.DataFrame(df_values, columns=[
                      "model", "prompt", "Mutation", "Code Difference"])
    MSE_table = []
    for llm in ["gpt-3.5-turbo", "gpt-4o"]:
        for prompt in [f"prompt{i}" for i in range(1, 12)]:
            df_temp = df[(df["model"] == llm) & (df["prompt"] == prompt)]
            if df_temp.empty:
                continue
            title1 = f"results/code_diff/{prompt}_{llm}_code-diff.png"
            title2 = f"results/code_diff/{prompt}_{llm}_ratio.png"
            violin_plot(df_temp, prompt, llm, title1)
            ratio_plot(df_temp, prompt, llm, title2)
            MSE_table += [MSE(df_temp)]
    for llm in ["gpt-3.5-turbo", "gpt-4o"]:
        for mutation in [2, 5, 10, 20, 40]:
            df_temp = df[(df["model"] == llm) & (df["Mutation"] == mutation)]
            if df_temp.empty:
                continue
            title1 = f"results/code_diff/{llm}_{mutation}_code-diff.png"
            title2 = f"results/code_diff/{llm}_{mutation}_ratio.png"
            violin_mutation_plot(df_temp, llm, mutation, title1)
            ratio_mutation_plot(df_temp, llm, mutation, title2)
    save_MSE_table(df)
    # return df
    #     title1 = f"results/code_diff/{model}_code-diff.png"
    #     title2 = f"results/code_diff/{model}_ratio.png"
    #     violin_plot(df, prompt, llm, title1)
    #     ratio_plot(df, prompt, llm, title2)
    #     MSE_table += [MSE(df)]
    # # save_MSE_table(MSE_table)
