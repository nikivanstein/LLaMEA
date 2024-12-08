import numpy as np

class NovelDynamicHarmonySearchAlgorithm:
    def __init__(self, budget, dim, harmony_memory_size=20, min_bandwidth=0.01, max_bandwidth=0.1, min_exp_prob=0.7, max_exp_prob=0.95):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth
        self.min_exp_prob = min_exp_prob
        self.max_exp_prob = max_exp_prob

    def initialize_harmony_memory(self):
        return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

    def improvise_harmony(self, harmony_memory, iteration):
        bandwidth = self.min_bandwidth + (self.max_bandwidth - self.min_bandwidth) * (iteration / self.budget)
        exp_prob = self.max_exp_prob - (self.max_exp_prob - self.min_exp_prob) * (iteration / self.budget)

        new_harmony = np.copy(harmony_memory[np.random.randint(self.harmony_memory_size)])
        for i in range(self.dim):
            if np.random.rand() < bandwidth:
                new_harmony[i] = np.random.uniform(-5.0, 5.0)
            if np.random.rand() < exp_prob:
                idxs = np.random.choice(self.harmony_memory_size, 3, replace=False)
                diff = harmony_memory[idxs[1]] - harmony_memory[idxs[2]]
                new_harmony[i] = harmony_memory[idxs[0]][i] + np.random.uniform(0, 1) * diff[i]
        return new_harmony

    def __call__(self, func):
        harmony_memory = self.initialize_harmony_memory()
        harmony_fitness = np.array([func(individual) for individual in harmony_memory])

        for iteration in range(self.budget):
            new_harmony = self.improvise_harmony(harmony_memory, iteration)
            new_fitness = func(new_harmony)
            worst_idx = np.argmax(harmony_fitness)
            
            if new_fitness < harmony_fitness[worst_idx]:
                harmony_memory[worst_idx] = new_harmony
                harmony_fitness[worst_idx] = new_fitness
        
        return harmony_memory[np.argmin(harmony_fitness)]