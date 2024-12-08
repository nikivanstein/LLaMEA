import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 20
        self.bandwidth = 0.01
        self.par = 0.7
        self.hm_acceptance_rate = 0.95
        self.max_imp = 1000
        
    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, size=(self.harmony_memory_size, self.dim))
        
        def adjust_pitch(value, lower, upper):
            new_value = value + np.random.uniform(-1, 1) * self.bandwidth
            return np.clip(new_value, lower, upper)
        
        def update_harmony_memory(harmony_memory, new_solution):
            fitness = func(new_solution)
            idx = np.argmax(harmony_memory[:, -1])
            if harmony_memory[idx, -1] > fitness:
                harmony_memory[idx, :-1] = new_solution
                harmony_memory[idx, -1] = fitness
            return harmony_memory
        
        harmony_memory = np.zeros((self.harmony_memory_size, self.dim + 1))
        harmony_memory[:, :-1] = initialize_harmony_memory()
        harmony_memory[:, -1] = np.array([func(sol) for sol in harmony_memory[:, :-1]])
        
        imp_count = 0
        while imp_count < self.max_imp:
            new_solution = np.array([adjust_pitch(hm, self.lower_bound, self.upper_bound) if np.random.rand() < self.par else hm for hm in harmony_memory[np.random.choice(self.harmony_memory_size)]])
            harmony_memory = update_harmony_memory(harmony_memory, new_solution)
            imp_count += 1
        
        best_solution = harmony_memory[np.argmin(harmony_memory[:, -1])]
        return best_solution[:-1]