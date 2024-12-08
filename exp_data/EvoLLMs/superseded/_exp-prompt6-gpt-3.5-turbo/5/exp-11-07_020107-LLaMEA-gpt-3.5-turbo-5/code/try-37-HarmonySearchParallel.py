import numpy as np
from multiprocessing import Pool

class HarmonySearchParallel:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hm_size = 20
        self.par_min = 0.4
        self.par_max = 0.9
        self.bw_min = 0.01
        self.bw_max = 0.1
        self.pitch_adj_prob = np.random.rand(self.dim)

    def __call__(self, func):
        def random_solution():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        
        def initialize_harmony_memory():
            return [random_solution() for _ in range(self.hm_size)]
        
        def pitch_adjustment(solution, hm):
            idx = np.random.randint(self.hm_size)
            r = np.random.uniform(0, 1, self.dim)
            mask = r < self.pitch_adj_prob
            new_solution = np.where(mask, solution, hm[idx])
            return new_solution
        
        def explore(solution):
            bw = np.random.uniform(self.bw_min, self.bw_max)
            rand_vec = np.random.uniform(-bw, bw, self.dim)
            new_solution = np.clip(solution + rand_vec, self.lower_bound, self.upper_bound)
            return new_solution
        
        harmony_memory = initialize_harmony_memory()
        
        def optimize(harmony):
            if np.random.rand() < 0.5:
                harmony = pitch_adjustment(harmony, harmony_memory)
            else:
                harmony = explore(harmony)
            
            if func(harmony) < func(harmony_memory[-1]):
                return harmony
            else:
                return harmony_memory[-1]
        
        with Pool() as p:
            for _ in range(self.budget):
                harmonies = [random_solution() for _ in range(self.hm_size)]
                updated_harmonies = p.map(optimize, harmonies)
                harmony_memory = sorted(updated_harmonies, key=lambda x: func(x))
        
        return harmony_memory[0]