import os
import json
import difflib
import seaborn as sns
import matplotlib.pyplot as plt


def check_exp_num(files):
    mutations = {"2": 0, "5": 0, "10": 0, "20": 0}
    for f in files:
        mutation = f.split("-")[-1]
        mutations[mutation] += 1
    print(mutations)


def check_same_code(files, exp_dir):
    countera = 0
    counterb = 0
    for f in files:
        codes = os.listdir(f"{exp_dir}/{f}/code/")
        if len(codes) != 2:
            continue
        with open(f"{exp_dir}/{f}/code/{codes[0]}", "r") as file:
            code1 = file.readlines()
        with open(f"{exp_dir}/{f}/code/{codes[1]}", "r") as file:
            code2 = file.readlines()
        if code_compare(code1, code2) == 0:
            countera += 1
            log_file = f"{exp_dir}/{f}/log.jsonl"
            with open(log_file, 'r', encoding='utf-8') as fread:
                lines = fread.readlines()
                line = lines[1]
                data = json.loads(line)
                if "No code was extracted." in data["feedback"]:
                    # print(data["feedback"])
                    counterb += 1
                    print(f"No code was extracted. {f}/conversationlog.jsonl")
                # else:
                #     print("==============")
                #     print(data["feedback"])
        else:
            print(f"{f}/conversationlog.jsonl")
            # log_data = []
            # with open(log_file
    print(counterb / countera)
            #         log_data += [data]
            # for data in log_data:
            #     print(data["feedback"])


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


def code_diff(files, exp_dir):
    diffs = {"2": [], "5": [], "10": [], "20": []}
    for f in files:
        codes = os.listdir(f"{exp_dir}/{f}/code/")
        if len(codes) != 2:
            continue
        with open(f"{exp_dir}/{f}/code/{codes[0]}", "r") as file:
            code1 = file.readlines()
        with open(f"{exp_dir}/{f}/code/{codes[1]}", "r") as file:
            code2 = file.readlines()
        diff_ratio = code_compare(code1, code2)
        if diff_ratio == 0:
            continue
        mutation = f.split("-")[-1]
        diffs[mutation].append(diff_ratio)
    for mutation in diffs:
        print(f"{len(diffs[mutation])} samples, mutation {mutation}: " +
              f"{sum(diffs[mutation]) / len(diffs[mutation])}")
    return diffs


def plot_diffs(diffs, exp_name):
    # draw violin plot of diffs for each mutation
    sns.violinplot(data=[diffs["2"], diffs["5"],
                   diffs["10"], diffs["20"]], cut=True)
    plt.xticks([0, 1, 2, 3], ["2", "5", "10", "20"])
    plt.savefig(f"{exp_name}-diffs.png")
    plt.clf()


for exp_dir in ["100-3.5", "100-4o"]:
    files = os.listdir(exp_dir)
    check_exp_num(files)
    check_same_code(files, exp_dir)
    diffs = code_diff(files, exp_dir)
    plot_diffs(diffs, exp_dir)
