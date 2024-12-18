import numpy as np

class EnhancedDynamicHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.par_min = 0.1
        self.par_max = 0.9
        self.bandwidth_min = 0.01
        self.bandwidth_max = 0.1

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))

        def update_parameters(iteration):
            par = self.par_min + (self.par_max - self.par_min) * (iteration / self.budget)
            bandwidth = self.bandwidth_min + (self.bandwidth_max - self.bandwidth_min) * (iteration / self.budget)
            return par, bandwidth

        def improvise_harmony(harmony_memory, par, bandwidth):
            new_harmony = np.copy(harmony_memory[np.random.randint(0, self.harmony_memory_size)])
            for i in range(self.dim):
                if np.random.uniform() < par:
                    new_harmony[i] += np.random.uniform(-bandwidth, bandwidth)
                    new_harmony[i] = np.clip(new_harmony[i], -5.0, 5.0)
            return new_harmony

        def apply_opposition(new_harmony):
            return 2.0 * np.mean(new_harmony) - new_harmony

        harmony_memory = initialize_harmony_memory()
        best_solution = None
        best_fitness = np.inf

        for iteration in range(self.budget):
            par, bandwidth = update_parameters(iteration)
            new_harmony = improvise_harmony(harmony_memory, par, bandwidth)
            new_harmony_opposite = apply_opposition(new_harmony)
            
            new_fitness = func(new_harmony)
            new_fitness_opposite = func(new_harmony_opposite)

            if new_fitness < best_fitness:
                best_solution = new_harmony
                best_fitness = new_fitness

            if new_fitness_opposite < best_fitness:
                best_solution = new_harmony_opposite
                best_fitness = new_fitness_opposite

            idx = np.argmax([func(h) for h in harmony_memory])
            if new_fitness < func(harmony_memory[idx]):
                harmony_memory[idx] = new_harmony

        return best_solution