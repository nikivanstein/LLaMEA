import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prob import TSPGLS, solve_instance_parallel_TSP
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
gap_data = pd.DataFrame(columns=["Algorithm", "Problem", "Scale", "Gap"])

# Function to append gaps to DataFrame
def append_gaps_to_dataframe(algorithm, problem, size, gap, gap_data):
    gap_data.loc[len(gap_data)] = {"Algorithm": algorithm, "Problem": problem, "Scale": size, "Gap": gap}
    return gap_data



goodalgs = ["EoH1-100", "EoH2-100", "EoH3-100",
    "EoH1", "EoH2", "EoH3", 
            
    #"AdaptiveVariationalPenaltyTSP", 
    #"DynamicAdaptivePenaltyTSP", "EnhancedDynamicEdgePenaltyPlusTSP", 
    "HeuristicAugmentedPenaltiesTSP", "RefinedDynamicEdgeMemoryPenaltyTSP", "RefinedDynamicEdgePenaltyPlusTSP"
    ]

labels = ["EoH-100-1", "EoH-100-2", "EoH-100-3", "EoH-2000-1", "EoH-2000-2", "EoH-2000-3", "LLaMEA-HPO-1", "LLaMEA-HPO-2", "LLaMEA-HPO-3"]



opt_costs = {
    "a280": 2579,
    "ali535": 202339,
    "att48": 10628,
    "att532": 27686,
    "bayg29": 1610,
    "bays29": 2020,
    "berlin52": 7542,
    "bier127": 118282,
    "brazil58": 25395,
    "brd14051": 469385,
    "brg180": 1950,
    "burma14": 3323,
    "ch130": 6110,
    "ch150": 6528,
    "d198": 15780,
    "d493": 35002,
    "d657": 48912,
    "d1291": 50801,
    "d1655": 62128,
    "d2103": 80450,
    "d15112": 1573084,
    "d18512": 645238,
    "dantzig42": 699,
    "dsj1000": 18660188,
    "eil51": 426,
    "eil76": 538,
    "eil101": 629,
    "fl417": 11861,
    "fl1400": 20127,
    "fl1577": 22249,
    "fl3795": 28772,
    "fnl4461": 182566,
    "fri26": 937,
    "gil262": 2378,
    "gr17": 2085,
    "gr21": 2707,
    "gr24": 1272,
    "gr48": 5046,
    "gr96": 55209,
    "gr120": 6942,
    "gr137": 69853,
    "gr202": 40160,
    "gr229": 134602,
    "gr431": 171414,
    "gr666": 294358,
    "hk48": 11461,
    "kroA100": 21282,
    "kroB100": 22141,
    "kroC100": 20749,
    "kroD100": 21294,
    "kroE100": 22068,
    "kroA150": 26524,
    "kroB150": 26130,
    "kroA200": 29368,
    "kroB200": 29437,
    "lin105": 14379,
    "lin318": 42029,
    "linhp318": 41345,
    "nrw1379": 56638,
    "p654": 34643,
    "pa561": 2763,
    "pcb442": 50778,
    "pcb1173": 56892,
    "pcb3038": 137694,
    "pla7397": 23260728,
    "pla33810": 66048945,
    "pla85900": 142382641,
    "pr76": 108159,
    "pr107": 44303,
    "pr124": 59030,
    "pr136": 96772,
    "pr144": 58537,
    "pr152": 73682,
    "pr226": 80369,
    "pr264": 49135,
    "pr299": 48191,
    "pr439": 107217,
    "pr1002": 259045,
    "pr2392": 378032,
    "rat99": 1211,
    "rat195": 2323,
    "rat575": 6773,
    "rat783": 8806,
    "rd100": 7910,
    "rd400": 15281,
    "rl1304": 252948,
    "rl1323": 270199,
    "rl1889": 316536,
    "rl5915": 565530,
    "rl5934": 556045,
    "rl11849": 923288,
    "si175": 21407,
    "si535": 48450,
    "si1032": 92650,
    "st70": 675,
    "swiss42": 1273,
    "ts225": 126643,
    "tsp225": 3916,
    "u159": 42080,
    "u574": 36905,
    "u724": 41910,
    "u1060": 224094,
    "u1432": 152970,
    "u1817": 57201,
    "u2152": 64253,
    "u2319": 234256,
    "ulysses16": 6859,
    "ulysses22": 7013,
    "usa13509": 19982859,
    "vm1084": 239297,
    "vm1748": 336556
}
path = os.path.dirname(os.path.abspath(__file__))
instances, instances_scale, instances_name = read_instance_all(path+"/tsplib/", 10000)
print(instances_name)

for alg_i in tqdm.tqdm(range(len(goodalgs))):
    alg = goodalgs[alg_i]
    #print(filename)
    h = alg
    prob = TSPGLS()
    prob.n_inst_eva = n_test_ins
    #print("Start evaluation...", h)

    # Create a ProcessPoolExecutor with the number of available CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
        # Submit tasks for parallel execution
        futures = [
            executor.submit(solve_instance_parallel_TSP, i, (opt_costs[instances_name[i]] / instances_scale[i]), instances[i], 
                            120, 1000, 
                            prob.perturbation_moves, f"llamea.tspalgs.{h}")
            for i in range(len(instances))
        ]

        # Collect the results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            gap, i = future.result()
            gap_data = append_gaps_to_dataframe(labels[alg_i], instances_name[i], instances_scale[i], gap, gap_data) 


# Save the DataFrame to a CSV and Pickle file
gap_data.to_csv("gap_data_TSP.csv", index=False)
gap_data.to_pickle("gap_data_TSP.pkl")

# Plot the violin plot using Seaborn
# plt.figure(figsize=(10, 6))
# sns.violinplot(x="Problem_Size", y="Gap", hue="Algorithm", data=gap_data, split=True)
# plt.xlabel("TSP Problem Size")
# plt.title("Gap Distribution by Algorithm and Problem")
# plt.savefig("tsplib_results.pdf")