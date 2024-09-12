import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prob import TSPGLS
import pickle
import time
import importlib
import os
import numpy as np
import tqdm
from utils.readTSPLib import read_instance_all
import concurrent

debug_mode = False
# problem_size = [10,20,50,100,200]
n_test_ins = 1000

# Initialize DataFrame to collect gaps
gap_data = pd.DataFrame(columns=["Algorithm", "Problem_Size", "Gap"])

# Function to append gaps to DataFrame
def append_gaps_to_dataframe(algorithm, size, gaps, gap_data):
    for gap in gaps:
        gap_data.loc[len(gap_data)] = {"Algorithm": algorithm, "Problem_Size": size, "Gap": gap}
    return gap_data



goodalgs = ["EoH1-100", "EoH2-100", "EoH3-100",
    "EoH1", "EoH2", "EoH3", 
            
    #"AdaptiveVariationalPenaltyTSP", 
    #"DynamicAdaptivePenaltyTSP", "EnhancedDynamicEdgePenaltyPlusTSP", 
    "HeuristicAugmentedPenaltiesTSP", "RefinedDynamicEdgeMemoryPenaltyTSP", "RefinedDynamicEdgePenaltyPlusTSP"
    ]

labels = ["EoH-100-1", "EoH-100-2", "EoH-100-3", "EoH-2000-1", "EoH-2000-2", "EoH-2000-3", "LLaMEA-HPO-1", "LLaMEA-HPO-2", "LLaMEA-HPO-3"]

path = os.path.dirname(os.path.abspath(__file__))
instances, instances_scale, instances_name = read_instance_all(path+"/tsplib/")
print(instances_name)

a = aaaa
i = 0
for alg in tqdm.tqdm(goodalgs):
    #print(filename)
    h = alg
    prob = TSPGLS()
    prob.n_inst_eva = n_test_ins
    #print("Start evaluation...", h)
    with open(f"results/TSP-{h}.txt", "w") as file:
         
        for problem_i in range(len(instances_name)):
            instance = instances[problem_i]
            scale = instances_scale[problem_i]
            name = instances_name[problem_i]

            gaps = np.zeros(self.n_inst_eva)

            # Create a ProcessPoolExecutor with the number of available CPU cores
            with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
                # Submit tasks for parallel execution
                futures = [
                    executor.submit(solve_instance_parallel, i, self.opt_costs[i], self.instances[i], 
                                    self.coords[i], self.time_limit, self.ite_max, 
                                    self.perturbation_moves, heuristic_name)
                    for i in range(self.n_inst_eva)
                ]

                # Collect the results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    gaps[i] = future.result()

            return gaps
            time_start = time.time()
            gaps = prob.testGLS(f"llamea.tspalgs.{h}", instance_dataset)

            result = (f"{h} - Average dis on {n_test_ins} instance with size {size} is: {np.mean(gaps):7.5f} timecost: {time.time()-time_start:7.3f}")
            #print(result)
            file.write(result + "\n")
            # Append gaps to DataFrame
            gap_data = append_gaps_to_dataframe(labels[i], size, gaps, gap_data) 
    i += 1


# for h in ["AdaptiveEdgePenaltyTSP", "DynamicPenaltyGradientOptimizer", "SynergisticAdaptivePenaltyRefinementPlus"]:

#     prob = TSPGLS()
#     prob.n_inst_eva = n_test_ins
#     print("Start evaluation...", h)
#     with open(f"results/{h}.txt", "w") as file:
#         for size in problem_size:
#             path = os.path.dirname(os.path.abspath(__file__))
#             instance_file_name = path+'/TestingData/TSP' + str(size)+ '.pkl'
#             with open(instance_file_name, 'rb') as f:
#                 instance_dataset = pickle.load(f)

#             heuristic_module = importlib.import_module(f"{h}")
#             heuristic = importlib.reload(heuristic_module)
            

#             time_start = time.time()
#             gaps = prob.testGLS(heuristic, instance_dataset)

#             result = (f"{h} - Average dis on {n_test_ins} instance with size {size} is: {np.mean(gaps):7.5f} timecost: {time.time()-time_start:7.3f}")
#             print(result)
#             file.write(result + "\n")
#             # Append gaps to DataFrame
#             gap_data = append_gaps_to_dataframe(h, size, gaps, gap_data)

# Save the DataFrame to a CSV and Pickle file
gap_data.to_csv("gap_data.csv", index=False)
gap_data.to_pickle("gap_data.pkl")

# Plot the violin plot using Seaborn
plt.figure(figsize=(10, 6))
sns.violinplot(x="Problem_Size", y="Gap", hue="Algorithm", data=gap_data, split=True)
plt.xlabel("TSP Problem Size")
plt.title("Gap Distribution by Algorithm and Problem Size")
plt.savefig("tsp_results.pdf")