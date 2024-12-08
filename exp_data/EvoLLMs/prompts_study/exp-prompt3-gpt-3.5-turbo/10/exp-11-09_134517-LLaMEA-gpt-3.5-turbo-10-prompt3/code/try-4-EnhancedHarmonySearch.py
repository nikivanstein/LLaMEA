import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def initialize_harmony():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def adjust_value(value):
            return np.clip(value, self.lower_bound, self.upper_bound)

        def harmony_search():
            harmony_memory = [initialize_harmony() for _ in range(self.budget)]
            best_solution = np.copy(harmony_memory[0])
            best_fitness = func(best_solution)
            bandwidth = 1.0

            for _ in range(self.budget):
                new_harmony = np.mean(harmony_memory, axis=0)
                new_harmony = adjust_value(new_harmony)
                new_fitness = func(new_harmony)

                if new_fitness < best_fitness:
                    best_solution = np.copy(new_harmony)
                    best_fitness = new_fitness

                index = np.random.randint(self.dim)
                perturbation = np.random.uniform(-bandwidth, bandwidth)
                new_harmony[index] = adjust_value(new_harmony[index] + perturbation)

                harmony_memory[np.argmax([func(h) for h in harmony_memory])] = new_harmony

                bandwidth *= 0.99  # Decaying bandwidth for exploration

            return best_solution

        return harmony_search()