#         if np.random.rand() < np.exp((fitness[idx] - new_fitness) / self.bandwidth):
# ```

# Original code:
import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.bandwidth = 0.5

    def __call__(self, func):
        def initialize_harmony_memory(HM_size):
            return np.random.uniform(self.lower_bound, self.upper_bound, (HM_size, self.dim))

        def adjust_bandwidth(iteration):
            return self.bandwidth * np.exp(-3 * iteration / self.budget)

        HM_size = 10
        HM = initialize_harmony_memory(HM_size)
        fitness = np.apply_along_axis(func, 1, HM)
        for i in range(self.budget):
            new_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            for j in range(self.dim):
                if np.random.rand() < 0.5:
                    new_solution[j] = HM[np.random.randint(HM_size), j]
            new_fitness = func(new_solution)
            if np.random.rand() < np.exp((fitness[idx] - new_fitness) / self.bandwidth):
                idx = np.argmax(fitness)
                HM[idx] = new_solution
                fitness[idx] = new_fitness
            self.bandwidth = adjust_bandwidth(i)
        best_idx = np.argmin(fitness)
        return HM[best_idx]