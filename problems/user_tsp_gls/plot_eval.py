import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the DataFrame from the pickle file
gap_data = pd.read_pickle("gap_data.pkl")

# Create a boxplot to compare the different algorithms
plt.figure(figsize=(10, 6))
sns.boxplot(x="Algorithm", y="Gap", data=gap_data)

# Add titles and labels
plt.title("Comparison of Gaps by Algorithm")
plt.xlabel("Algorithm")
plt.ylabel("Gap")

# Show the plot
plt.xticks(rotation=45)  # Rotate x-axis labels if needed for better readability
plt.tight_layout()
plt.savefig("tsp_results.pdf")
plt.clf()

plt.figure(figsize=(14, 6))
sns.boxplot(x="Algorithm", y="Gap", hue="Problem_Size", data=gap_data,  dodge=True, linewidth=0.2)

# Add titles and labels
plt.title("Comparison of Gaps by Algorithm and Problem Size")
plt.xlabel("Algorithm")
plt.ylabel("Gap")

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)
#plt.yscale("symlog")
# Adjust the layout to make it look better
plt.tight_layout()
plt.savefig("tsp_results2.pdf")


plt.clf()

plt.figure(figsize=(16, 8))
g = sns.FacetGrid(gap_data, col="Problem_Size", height=5, aspect=1.5)
g.map(sns.boxplot, "Algorithm", "Gap")

# Rotate x-axis labels for better readability
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# Add titles and labels
g.fig.suptitle("Comparison of Gaps by Algorithm and Problem Size", y=1.05)
plt.tight_layout()
plt.savefig("tsp_results3.pdf")

plt.clf()
plt.figure(figsize=(14, 7))
sns.barplot(x="Algorithm", y="Gap", hue="Problem_Size", data=gap_data, ci="sd", dodge=True)

# Add titles and labels
plt.title("Comparison of Gaps by Algorithm and Problem Size")
plt.xlabel("Algorithm")
plt.ylabel("Gap")

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Adjust the layout to make it look better
plt.tight_layout()
plt.savefig("tsp_results4.pdf")