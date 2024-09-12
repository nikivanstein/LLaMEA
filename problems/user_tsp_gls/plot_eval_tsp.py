import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the DataFrame from the pickle file
gap_data = pd.read_pickle("gap_data_TSP.pkl")

print(gap_data)

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
plt.savefig("tsplib.pdf")
plt.clf()