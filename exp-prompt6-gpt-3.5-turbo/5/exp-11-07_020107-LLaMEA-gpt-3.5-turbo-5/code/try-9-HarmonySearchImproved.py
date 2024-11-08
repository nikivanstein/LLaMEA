import numpy as np

class HarmonySearchImproved:
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

    def __call__(self, func):
        def random_solutions(num):
            return np.random.uniform(self.lower_bound, self.upper_bound, (num, self.dim))
        
        def initialize_harmony_memory():
            return random_solutions(self.hm_size)
        
        def pitch_adjustment(solution, hm):
            idx = np.random.randint(self.hm_size)
            r = np.random.uniform(0, 1, (self.dim,))
            mask = np.where(r < self.par, 1, 0)
            new_solution = solution * mask + hm[idx] * (1 - mask)
            return new_solution
        
        def explore(solution):
            bw = np.random.uniform(self.bw_min, self.bw_max)
            rand_vec = np.random.uniform(-bw, bw, (self.dim,))
            new_solution = np.clip(solution + rand_vec, self.lower_bound, self.upper_bound)
            return new_solution
        
        harmony_memory = initialize_harmony_memory()
        for _ in range(self.budget):
            harmonies = random_solutions(self.budget)
            masks = np.random.uniform(0, 1, (self.budget, self.dim))
            pitch_mask = masks < 0.5
            harmony_mask = np.logical_not(pitch_mask)
            harmony = harmonies * harmony_mask + harmony_memory[-1] * pitch_mask
            harmony = np.where(np.random.rand(self.budget, 1) < 0.5, pitch_adjustment(harmony, harmony_memory), explore(harmony))
            
            costs = np.apply_along_axis(func, 1, harmony)
            min_idx = np.argmin(costs)
            if costs[min_idx] < func(harmony_memory[-1]):
                harmony_memory[-1] = harmony[min_idx]
                harmony_memory = sorted(harmony_memory, key=lambda x: func(x))
        
        return harmony_memory[0]