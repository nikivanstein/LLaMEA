import numpy as np

class HarmonySearch:
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
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.hm_size, self.dim))
        
        def pitch_adjustment(solution, hm):
            idx = np.random.randint(self.hm_size)
            r = np.random.uniform(0, 1, (self.dim,))
            mask = r < self.par
            new_solution = np.where(mask[:, None], solution, hm[idx])
            return new_solution
        
        def explore(solution):
            bw = np.random.uniform(self.bw_min, self.bw_max)
            rand_vec = np.random.uniform(-bw, bw, (self.dim,))
            new_solution = np.clip(solution + rand_vec, self.lower_bound, self.upper_bound)
            return new_solution
        
        harmony_memory = initialize_harmony_memory()
        for _ in range(self.budget):
            harmony = np.random.uniform(self.lower_bound, self.upper_bound, (self.dim,))
            if np.random.rand() < 0.5:
                harmony = pitch_adjustment(harmony, harmony_memory)
            else:
                harmony = explore(harmony)
            
            idx = np.argmin([func(h) for h in harmony_memory])
            if func(harmony) < func(harmony_memory[idx]):
                harmony_memory[idx] = harmony
        
        return harmony_memory[np.argmin([func(h) for h in harmony_memory])]