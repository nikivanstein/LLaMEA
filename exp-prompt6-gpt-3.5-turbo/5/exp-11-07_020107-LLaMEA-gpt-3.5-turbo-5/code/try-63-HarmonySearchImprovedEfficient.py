import numpy as np

class HarmonySearchImprovedEfficient:
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

        harmony_memory = [random_solution() for _ in range(self.hm_size)]
        
        for _ in range(self.budget):
            harmony = random_solution()
            
            if np.random.rand() < 0.5:
                idx = np.random.randint(self.hm_size)
                r = np.random.uniform(0, 1, self.dim)
                mask = r < self.pitch_adj_prob
                harmony = np.where(mask, harmony, harmony_memory[idx])
            else:
                bw = np.random.uniform(self.bw_min, self.bw_max)
                harmony += np.random.uniform(-bw, bw, self.dim)
                harmony = np.clip(harmony, self.lower_bound, self.upper_bound)
            
            idx_to_replace = np.argmax([func(h) for h in harmony_memory])
            if func(harmony) < func(harmony_memory[idx_to_replace]):
                harmony_memory[idx_to_replace] = harmony
        
        return min(harmony_memory, key=func)