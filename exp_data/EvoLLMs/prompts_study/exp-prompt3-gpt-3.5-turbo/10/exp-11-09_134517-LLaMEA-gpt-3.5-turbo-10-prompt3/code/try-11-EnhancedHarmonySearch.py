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

        def adaptive_pitch_adjustment(value, pitch_adjustment_rate):
            return value + np.random.uniform(-pitch_adjustment_rate, pitch_adjustment_rate, self.dim)

        def harmony_search():
            harmony_memory = [initialize_harmony() for _ in range(self.budget)]
            best_solution = np.copy(harmony_memory[0])
            best_fitness = func(best_solution)
            pitch_adjustment_rate = 0.1

            for _ in range(self.budget):
                new_harmony = np.mean(harmony_memory, axis=0)
                new_harmony = adjust_value(new_harmony)
                new_fitness = func(new_harmony)

                if new_fitness < best_fitness:
                    best_solution = np.copy(new_harmony)
                    best_fitness = new_fitness

                index = np.random.randint(self.dim)
                pitch_adjustment_rate *= 0.95  # Adaptive pitch adjustment
                new_harmony[index] = adaptive_pitch_adjustment(new_harmony, pitch_adjustment_rate)
                new_harmony = adjust_value(new_harmony)

                new_harmony_opposite = adjust_value(self.lower_bound + self.upper_bound - new_harmony)
                new_fitness_opposite = func(new_harmony_opposite)

                if new_fitness_opposite < best_fitness:
                    best_solution = np.copy(new_harmony_opposite)
                    best_fitness = new_fitness_opposite

                harmony_memory[np.argmax([func(h) for h in harmony_memory])] = new_harmony

            return best_solution

        return harmony_search()