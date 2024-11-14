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
                if code_diff_value <= 0.001:
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
    plt.yscale("log")
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
    weights = [0.7219028, 0.18262857, 0.06456895, 0.02282857, 0.00807112]
    MSE_value = [llm, prompt]
    for mutation in [2, 5, 10, 20, 40]:
        df_temp = df[df["Mutation"] == mutation]
        if df_temp["Code Difference"].values.all() == 0:
            MSE_value += [np.nan]
        else:
            MSE_value += [((np.log10(df_temp["Code Difference"]
                           * 100/mutation))**2).mean()]
    MSE_value += [np.average(MSE_value[2:], weights=weights)]
    return MSE_value


def save_MSE_table(df):
    MSE_table = []
    for llm in ["gpt-3.5-turbo", "gpt-4o"]:
        for prompt in [f"prompt{i}" for i in range(1, 12)]:
            df_temp = df[(df["model"] == llm) & (df["prompt"] == prompt)]
            if df_temp.empty:
                continue
            MSE_table += [MSE(df_temp, llm, prompt)]
    MSE_table = pd.DataFrame(MSE_table, columns=["model", "prompt", "2%", "5%",
                                                 "10%", "20%", "40%", "score"])
    MSE_table.to_csv("results/code_diff/MSE_table.csv")
    df_gpt_3_5 = MSE_table[MSE_table["model"] == "gpt-3.5-turbo"]
    df_gpt_4 = MSE_table[MSE_table["model"] == "gpt-4o"]
    plt.figure(figsize=(4.2, 4))
    sns.heatmap(df_gpt_3_5.iloc[:, 2:], annot=True, fmt=".2f", square=True,
                cmap="gist_yarg", yticklabels=df_gpt_3_5["prompt"],
                xticklabels=["2%", "5%", "10%", "20%", "40%", "score"])
    plt.tight_layout()
    plt.savefig("results/code_diff/MSE_gpt-3.5-turbo.png")
    plt.clf()
    plt.figure(figsize=(4.2, 4))
    sns.heatmap(df_gpt_4.iloc[:, 2:], annot=True, fmt=".2f", square=True,
                cmap="gist_yarg", yticklabels=df_gpt_4["prompt"],
                xticklabels=["2%", "5%", "10%", "20%", "40%", "score"])
    plt.tight_layout()
    plt.savefig("results/code_diff/MSE_gpt-4o.png")
    plt.clf()


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
    plt.yscale("log")
    plt.xlabel("different prompts")
    plt.ylabel("ratio")
    plt.title(
        f"Ratio of Delivered Code Difference to Requested Mutation Rate\nwhen Requesting {mutation}% Mutation Rate with {llm}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(title)
    plt.cla()


def aggregate_plot(df):
    cud = ["#e69f00", "#56b4e9", "#009e73", "#f0e442",
           "#0072b2", "#d55e00", "#cc79a7", "#000000"]
    df["aggregate_label"] = df["prompt"] + \
        "_" + df["Mutation"].astype(str) + "%"
    df["prompt_number"] = df["prompt"].str[6:].astype(int)
    df["ratio"] = (df["Code Difference"] * 100) / df["Mutation"]
    df_sorted = df.sort_values(
        by=["Mutation", "prompt_number"], ascending=[True, True])
    for model in ["gpt-3.5-turbo", "gpt-4o"]:
        xticks = [i for i in range(8)]
        for i in range(4):
            xticks += [j+9*(i+1) for j in range(8)]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
        df_temp = df_sorted[df_sorted["model"] == model]
        # plt.figure(figsize=(12, 4))
        sns.violinplot(x="aggregate_label", y="Code Difference", data=df_temp,
                       cut=0, inner="box", palette=cud, hue="prompt",
                       inner_kws=dict(box_width=3), legend=True, ax=ax1)
        positions = range(len(df_temp["aggregate_label"].unique()))
        for y in [0.02, 0.05, 0.1, 0.2, 0.4]:
            i = [0.02, 0.05, 0.1, 0.2, 0.4].index(y)
            ax1.hlines(y=y, xmin=positions[8*i]-0.2, xmax=positions[8*i+7]+0.2,
                       color='red', linestyles="dashed", linewidth=2)
            if y != 0.4:
                ax1.axvline(x=positions[8*i+7]+0.5, color='grey',
                            linestyle="dotted", linewidth=2)
        ax1.plot(0, 0, color='red', linestyle="dashed",
                 linewidth=1, label="requested mutation rate")
        ax1.set_ylabel("code difference")
        ax1.set_xlabel("")
        ax1.xaxis.tick_bottom()
        ax1.xaxis.set_label_position('bottom')
        ax1.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        # ax1.set_title(f"Code Difference Distribution of Different Prompts and Mutation Rates when Using {model}")
        ax1.legend(ncol=3)

        sns.stripplot(x="aggregate_label", y="ratio", data=df_temp, jitter=True,
                      palette=cud, hue="prompt", legend=True, ax=ax2)
        ax2.axhline(y=1, color='green', linestyle='-.', linewidth=2,
                    label="delivered code difference = requested mutation rate")
        for y in [0.02, 0.05, 0.1, 0.2]:
            i = [0.02, 0.05, 0.1, 0.2].index(y)
            ax2.axvline(x=positions[8*i+7]+0.5, color='grey',
                        linestyle="dotted", linewidth=2)
        ax2.set_yscale("log")
        ax2.set_ylabel("ratio")
        ax2.set_xlabel("")
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')
        # ax2.set_title(
        #     f"Ratio of Delivered Code Difference to Requested Mutation Rate of Different Prompts and Mutation Rates when Using {model}", pad=-30)
        ax2.legend(ncol=3, loc="upper right")
        # ax1.title(
        #     f"Code Difference Distribution of Different Prompts when Using {model}")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"results/code_diff/aggregate_diff_{model}.png")
        plt.clf()


def convergence_plot():
    cmap = plt.get_cmap('viridis')
    plt.figure(figsize=(8, 6))
    x = np.arange(budegt)
    cud = ["#e69f00", "#56b4e9", "#009e73", "#f0e442",
           "#0072b2", "#d55e00", "#cc79a7", "#000000"]
    colors = [cud[0], cud[1], cud[2], cud[0], cud[1], cud[2]]
    line_styles = ['solid', 'dashed', 'dashdot', 'solid', 'dashed', 'dashdot']
    labels = [
        "1+1 GPT-4o-ES with prompt5", "1+1 GPT-4o-ES with prompt9",
        "1+1 GPT-4o-ES baseline", "1+1 GPT-3.5-ES with prompt2",
        "1+1 GPT-3.5-ES with prompt7", "1+1 GPT-3.5-ES baseline",
    ]
    i = 0
    for model in ["gpt-4o", "gpt-3.5-turbo"]:
        for label in ["exp1", "exp2", "baseline"]:
            mean_aucs = np.loadtxt(f"results/{model}-{label}-aucs.txt")
            std_aucs = np.loadtxt(f"results/{model}-{label}-std.txt")
            if label != "baseline":
                std_aucs = std_aucs**0.5
            plt.plot(x, mean_aucs, color=colors[i], linewidth=2,
                     linestyle=line_styles[i], label=labels[i])
            plt.fill_between(x, mean_aucs - std_aucs, mean_aucs +
                             std_aucs, color=colors[i], alpha=0.05)
            plt.xlim(0, 100)
            i += 1
        plt.xlabel("Iterations")
        plt.ylabel("mean AOCC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/convergence_{model}.png")
        plt.clf()


if __name__ == "__main__":
    # models = ["test", "test2", "test3-gpt-3.5-turbo", "test3-gpt-4o",
    #           "gpt-4o", "gpt-3.5-turbo",
    #           "Llama-3.2-1B", "Llama-3.2-3B",
    #           "Meta-Llama-3.1-8B", "Meta-Llama-3.1-70B"]
    models = [f"prompt{i}_{m}" for m in ["gpt-3.5-turbo", "gpt-4o"]
              for i in range(1, 12)]
    mutations = ["beta1.5", "2"] + [str(i*5) for i in range(1, 11)]
    df_values = []
    print("Building data...")
    # for model in models:
    #     prompt = model.split("_")[0]
    #     llm = model.split("_")[1]
    #     exp_name = f"{prompt}-{llm}"
    #     exp_dirs = []
    #     labels = []
    #     if not os.path.exists(f"exp-{exp_name}"):
    #         continue
    #     for mutation in mutations:
    #         if not os.path.exists(f"exp-{exp_name}/{mutation}"):
    #             continue
    #         folders = os.listdir(
    #             f"exp-{exp_name}/{mutation}")
    #         exp_dirs += [
    #             [f"exp-{exp_name}/{mutation}/{f}" for f in folders if f.startswith("exp")]]
    #         labels += [f"{mutation}"]
    #     df_values += build_data(exp_dirs, prompt, llm, labels)
    # df = pd.DataFrame(df_values, columns=[
    #                   "model", "prompt", "Mutation", "Code Difference"])
    # df.to_csv("results/code_diff/code_diff.csv")
    df = pd.read_csv("results/code_diff/code_diff.csv")
    print("Saving MSE table...")
    save_MSE_table(df.copy())
    print("Plotting aggregation...")
    aggregate_plot(df.copy())
    print("Plotting convergence...")
    convergence_plot()
    # for llm in ["gpt-3.5-turbo", "gpt-4o"]:
    #     for prompt in [f"prompt{i}" for i in range(1, 12)]:
    #         df_temp = df[(df["model"] == llm) & (df["prompt"] == prompt)]
    #         if df_temp.empty:
    #             continue
    #         title1 = f"results/code_diff/{prompt}_{llm}_code-diff.png"
    #         title2 = f"results/code_diff/{prompt}_{llm}_ratio.png"
    #         print(f"Plotting {prompt} with {llm}...")
    #         violin_plot(df_temp, prompt, llm, title1)
    #         ratio_plot(df_temp, prompt, llm, title2)
    # for llm in ["gpt-3.5-turbo", "gpt-4o"]:
    #     for mutation in [2, 5, 10, 20, 40]:
    #         df_temp = df[(df["model"] == llm) & (df["Mutation"] == mutation)]
    #         if df_temp.empty:
    #             continue
    #         title1 = f"results/code_diff/{llm}_{mutation}_code-diff.png"
    #         title2 = f"results/code_diff/{llm}_{mutation}_ratio.png"
    #         print(f"Plotting {mutation}% with {llm}...")
    #         violin_mutation_plot(df_temp, llm, mutation, title1)
    #         ratio_mutation_plot(df_temp, llm, mutation, title2)
