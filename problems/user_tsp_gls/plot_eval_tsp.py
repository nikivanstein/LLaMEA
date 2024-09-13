import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import matplotlib.pyplot as plt
import seaborn as sns
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#matplotlib.rcParams['font.family'] = 'sans-serif'
#matplotlib.rcParams['font.sans-serif'] = 'Arial'

import operator
import math
import networkx

def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, labels=False, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
        labels (bool, optional): if set to `True`, the calculated avg rank
        values will be displayed
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=2)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom", size=16)

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=linewidth)
        if labels:
            text(textspace + 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="right", va="center", size=10)
        text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=16)

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=linewidth)
        if labels:
            text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="left", va="center", size=10)
        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
             ha="left", va="center", size=16)

    # no-significance lines
    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2

        for l, r in lines:
            line([(rankpos(ssums[l]) - side, start),
                  (rankpos(ssums[r]) + side, start)],
                 linewidth=linewidth_sign)
            start += height
            print('drawing: ', l, r)

    # draw_lines(lines)
    start = cline + 0.2
    side = -0.02
    height = 0.1

    # draw no significant lines
    # get the cliques
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    print(nnames)
    for clq in cliques:
        if len(clq) == 1:
            continue
        print(clq)
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        line([(rankpos(ssums[min_idx]) - side, start),
              (rankpos(ssums[max_idx]) + side, start)],
             linewidth=linewidth_sign)
        start += height


def form_cliques(p_values, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)


def draw_cd_diagram(df_perf=None, alpha=0.05, title=None, labels=False):
    """
    Draws the critical difference diagram given the list of pairwise classifiers that are
    significant or not
    """
    p_values, average_ranks, _ = wilcoxon_holm(df_perf=df_perf, alpha=alpha)

    print(average_ranks)

    for p in p_values:
        print(p)


    graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
                cd=None, reverse=True, width=9, textspace=1.5, labels=labels)

    font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 22,
        }
    if title:
        plt.title(title,fontdict=font, y=0.9, x=0.5)
    plt.savefig('cd-diagram.png',bbox_inches='tight')

def wilcoxon_holm(alpha=0.05, df_perf=None):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis
    """
    print(pd.unique(df_perf['Algorithm']))
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['Algorithm']).size()}).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts['count'].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['Algorithm'])
    # test the null hypothesis using friedman before doing a post-hoc analysis
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['Algorithm'] == c]['Gap'])
        for c in classifiers))[1]
    if friedman_p_value >= alpha:
        # then the null hypothesis over the entire classifiers cannot be rejected
        print('the null hypothesis over the entire classifiers cannot be rejected')
        exit()
    # get the number of classifiers
    m = len(classifiers)
    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(m - 1):
        # get the name of classifier one
        classifier_1 = classifiers[i]
        # get the performance of classifier one
        perf_1 = np.array(df_perf.loc[df_perf['Algorithm'] == classifier_1]['Gap']
                          , dtype=np.float64)
        for j in range(i + 1, m):
            # get the name of the second classifier
            classifier_2 = classifiers[j]
            # get the performance of classifier one
            perf_2 = np.array(df_perf.loc[df_perf['Algorithm'] == classifier_2]
                              ['Gap'], dtype=np.float64)
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
            # appen to the list
            p_values.append((classifier_1, classifier_2, p_value, False))
    # get the number of hypothesis
    k = len(p_values)
    # sort the list in acsending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            # stop
            break
    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[df_perf['Algorithm'].isin(classifiers)]. \
        sort_values(['Algorithm', 'Problem'])
    # get the rank data
    rank_data = np.array(sorted_df_perf['Gap']).reshape(m, max_nb_datasets)

    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), columns=
    np.unique(sorted_df_perf['Problem']))

    # number of wins
    dfff = df_ranks.rank(ascending=True)
    print(dfff[dfff == 1.0].sum(axis=1))

    # average the ranks
    average_ranks = df_ranks.rank(ascending=True).mean(axis=1).sort_values(ascending=True)
    # return the p-values and the average ranks
    return p_values, average_ranks, max_nb_datasets



# Load the DataFrame from the pickle file
gap_data = pd.read_pickle("gap_data_TSP.pkl")

# Replace negative values with 0.000
gap_data['Gap'] = gap_data['Gap'].apply(lambda x: 0.000 if x < 0 else x)

# Pivot the DataFrame to have problems as rows and algorithms as columns
pivot_df = gap_data.pivot_table(index='Problem', columns='Algorithm', values='Gap', aggfunc='first')

# Round the values to 3 decimal places
pivot_df = pivot_df.round(3)

# Perform the Friedman test to compare multiple algorithms
stat, p_value = friedmanchisquare(*[pivot_df[alg] for alg in pivot_df.columns])

print(f"Friedman test statistic: {stat}, p-value: {p_value}")

if p_value < 0.05:
    print("There is a significant difference between the algorithms.")
    
    # Perform post-hoc Nemenyi test for pairwise comparisons
    nemenyi_results = posthoc_nemenyi_friedman(pivot_df.values)

    # Create a heatmap of the Nemenyi test results
    plt.figure(figsize=(10, 8))
    sns.heatmap(nemenyi_results, annot=True, cmap="coolwarm", xticklabels=pivot_df.columns, yticklabels=pivot_df.columns)
    plt.title("Nemenyi Post-hoc Test Results (P-values)")
    plt.tight_layout()
    plt.savefig("tsp_Nemenyi.png")
    
    print("Nemenyi post-hoc test results:")
    print(nemenyi_results)

    # Visualize the Critical Difference (CD) Diagram
    average_ranks = pivot_df.rank(axis=1, method='average').mean(axis=0)
    algorithms = list(pivot_df.columns)

    # Calculate the Critical Difference (CD)
    k = len(average_ranks)  # number of algorithms
    n = len(pivot_df)  # number of problems
    q_alpha = 2.291  # for alpha=0.05 and k <= 10
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    print(f"Critical Difference (CD): {cd:.3f}")
    
    # Plot the CD diagram
    plt.figure(figsize=(10, 6))
    
    # Sort algorithms by average rank
    sorted_algorithms = average_ranks.sort_values()
    
    # Plot each algorithm's rank as a point
    for i, (algorithm, rank) in enumerate(sorted_algorithms.items()):
        plt.plot([rank], [i], 'o', label=algorithm, markersize=10, color='black')
        plt.text(rank, i, f"{algorithm} ({rank:.3f})", va='center', ha='right', fontsize=10)
    
    # Draw horizontal lines for non-significant differences
    # Algorithms that are within the critical difference range
    for i, (alg_i, rank_i) in enumerate(sorted_algorithms.items()):
        for j, (alg_j, rank_j) in enumerate(sorted_algorithms.items()):
            if abs(rank_i - rank_j) <= cd and i != j:
                plt.plot([rank_i, rank_j], [i, j], color='gray', linewidth=2)

    # Draw the critical difference bar
    min_rank = sorted_algorithms.min()
    max_rank = min_rank + cd
    plt.plot([min_rank, max_rank], [len(sorted_algorithms)] * 2, lw=3, color='red')
    plt.text((min_rank + max_rank) / 2, len(sorted_algorithms), 'Critical Difference', va='center', ha='center', color='red', fontsize=12)
    
    # Set labels and title
    plt.title('Critical Difference Diagram')
    plt.xlabel('Average Rank')
    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    plt.grid(True)
    
    # Show the plot
    plt.tight_layout()
    plt.savefig("tsp_cdd.png")

    p_values, average_ranks, _ = wilcoxon_holm(df_perf=gap_data, alpha=0.05)

    graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
                cd=None, reverse=False, width=9, textspace=1.5, labels=True)

    font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 22,
        }
    #if title:
    #    plt.title(title,fontdict=font, y=0.9, x=0.5)
    plt.savefig('cd-diagram.png',bbox_inches='tight')
else:
    print("No significant difference between the algorithms.")

# Create a function to highlight the minimum value in bold
def highlight_min_in_latex(df):
    formatted_df = df.copy()
    for idx, row in df.iterrows():
        min_val = row.min()
        formatted_row = row.apply(lambda x: f"\\textbf{{{x:.3f}}}" if x == min_val else f"{x:.3f}")
        formatted_df.loc[idx] = formatted_row
    return formatted_df

# Apply the function to highlight the minimum value in each row
formatted_pivot_df = highlight_min_in_latex(pivot_df)

# Add an extra row showing the average score of each algorithm
average_row = pivot_df.mean().round(3)
print(average_row)
#average_row.name = 'Average'
#formatted_pivot_df = formatted_pivot_df.append(average_row.apply(lambda x: f"{x:.3f}"))

# Convert the modified DataFrame to a LaTeX table
latex_code = formatted_pivot_df.to_latex(escape=False, index=True)

# Store the LaTeX code in a file
with open('algorithm_problem_table.tex', 'w') as f:
    f.write(latex_code)

print("LaTeX table with averages has been saved.")
