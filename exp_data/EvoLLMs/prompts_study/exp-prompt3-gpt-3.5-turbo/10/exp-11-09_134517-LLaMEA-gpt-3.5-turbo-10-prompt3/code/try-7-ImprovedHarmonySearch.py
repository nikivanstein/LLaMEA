import numpy as np

class ImprovedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hm_accept_rate = 0.95

    def __call__(self, func):
        def initialize_harmony():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def adjust_value(value):
            return np.clip(value, self.lower_bound, self.upper_bound)

        def pitch_adjustment(iteration):
            return np.random.uniform(0, 1) * ((self.upper_bound - self.lower_bound) * (1 - iteration / self.budget))

        def harmony_search():
            harmony_memory = [initialize_harmony() for _ in range(self.budget)]
            best_solution = np.copy(harmony_memory[0])
            best_fitness = func(best_solution)

            for iteration in range(self.budget):
                new_harmony = np.mean(harmony_memory, axis=0)
                new_harmony = adjust_value(new_harmony)

                for i in range(self.dim):
                    if np.random.rand() < self.hm_accept_rate:
                        new_harmony[i] += pitch_adjustment(iteration)

                new_fitness = func(new_harmony)

                if new_fitness < best_fitness:
                    best_solution = np.copy(new_harmony)
                    best_fitness = new_fitness

                harmony_memory[np.argmax([func(h) for h in harmony_memory])] = new_harmony

            return best_solution

        return harmony_search()