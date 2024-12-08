import numpy as np

class ElitePreservationHarmonySearch:
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

        def opposition_based_learning(value):
            return self.lower_bound + self.upper_bound - value

        def harmony_search():
            harmony_memory = [initialize_harmony() for _ in range(self.budget)]
            best_solution = np.copy(harmony_memory[0])
            best_fitness = func(best_solution)

            for _ in range(self.budget):
                new_harmony = np.mean(harmony_memory, axis=0)
                new_harmony = adjust_value(new_harmony)
                new_fitness = func(new_harmony)

                if new_fitness < best_fitness:
                    best_solution = np.copy(new_harmony)
                    best_fitness = new_fitness

                index = np.random.randint(self.dim)
                new_harmony[index] = np.random.uniform(self.lower_bound, self.upper_bound)

                new_harmony_opposite = opposition_based_learning(new_harmony)
                new_fitness_opposite = func(new_harmony_opposite)

                if new_fitness_opposite < best_fitness:
                    best_solution = np.copy(new_harmony_opposite)
                    best_fitness = new_fitness_opposite

                # Introducing elite preservation
                elite_idx = np.argsort([func(h) for h in harmony_memory])[:int(0.1 * self.budget)]
                harmony_memory[elite_idx] = new_harmony

            return best_solution

        return harmony_search()