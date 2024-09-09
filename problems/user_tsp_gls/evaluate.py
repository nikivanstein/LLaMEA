from prob import TSPGLS
import pickle
import time
import importlib
import os

debug_mode = False
# problem_size = [10,20,50,100,200]
problem_size = [20,50,100]
n_test_ins = 1000

for h in ["EoH", "llamea"]:

    prob = TSPGLS()
    prob.n_inst_eva = n_test_ins
    print("Start evaluation...", h)
    with open(f"results-{h}.txt", "w") as file:
        for size in problem_size:
            path = os.path.dirname(os.path.abspath(__file__))
            instance_file_name = path+'/TestingData/TSP' + str(size)+ '.pkl'
            with open(instance_file_name, 'rb') as f:
                instance_dataset = pickle.load(f)

            heuristic_module = importlib.import_module(f"heuristic_{h}")
            heuristic = importlib.reload(heuristic_module)
            

            time_start = time.time()
            gap = prob.testGLS(heuristic, instance_dataset)

            result = (f"{h} - Average dis on {n_test_ins} instance with size {size} is: {gap:7.3f} timecost: {time.time()-time_start:7.3f}")
            print(result)
            file.write(result + "\n")
            
