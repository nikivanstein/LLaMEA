import numpy as np

class EnhancedDynamicPitchHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pitch_range = 0.1
        self.pitch_adapt_rate = 0.1

    def __call__(self, func):
        def initialize_harmony():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def adjust_value(value):
            return np.clip(value, self.lower_bound, self.upper_bound)

        def adjust_pitch(pitch, improvement):
            return max(0.001, pitch * np.exp(self.pitch_adapt_rate * improvement))

        def local_search(harmony, func_value):
            new_harmony = np.copy(harmony)
            for i in range(self.dim):
                original_value = new_harmony[i]
                new_harmony[i] = np.random.uniform(max(self.lower_bound, original_value - self.pitch_range),
                                                   min(self.upper_bound, original_value + self.pitch_range))
                if func(new_harmony) < func_value:
                    harmony[i] = new_harmony[i]  # Update if better
                else:
                    new_harmony[i] = original_value  # Revert if not better
            return new_harmony

        def opposition_based_learning(value):
            return self.lower_bound + self.upper_bound - value

        def harmony_search():
            harmony_memory = [initialize_harmony() for _ in range(self.budget)]
            best_solution = np.copy(harmony_memory[0])
            best_fitness = func(best_solution)
            pitch = self.pitch_range

            for _ in range(self.budget):
                new_harmony = np.mean(harmony_memory, axis=0)
                new_harmony = adjust_value(new_harmony)
                new_fitness = func(new_harmony)

                if new_fitness < best_fitness:
                    best_solution = np.copy(new_harmony)
                    best_fitness = new_fitness
                    pitch = adjust_pitch(pitch, 1)

                for i in range(self.dim):
                    harmony_memory[i] = opposition_based_learning(harmony_memory[i])

                local_best_harmony = local_search(new_harmony, new_fitness)

                for i in range(self.budget):
                    harmony_memory[i] = opposition_based_learning(harmony_memory[i])

                new_harmony_opposite = opposition_based_learning(local_best_harmony)
                new_fitness_opposite = func(new_harmony_opposite)

                if new_fitness_opposite < best_fitness:
                    best_solution = np.copy(new_harmony_opposite)
                    best_fitness = new_fitness_opposite
                    pitch = adjust_pitch(pitch, 1)

                harmony_memory[np.argmax([func(h) for h in harmony_memory])] = new_harmony

            return best_solution

        return harmony_search()