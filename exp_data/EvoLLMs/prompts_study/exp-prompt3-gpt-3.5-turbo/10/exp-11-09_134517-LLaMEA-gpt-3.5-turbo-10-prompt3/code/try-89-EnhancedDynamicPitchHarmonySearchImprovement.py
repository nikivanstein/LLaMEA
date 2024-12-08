import numpy as np

class EnhancedDynamicPitchHarmonySearchImprovement:
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
                # Incorporating Differential Evolution
                candidate = new_harmony[i] + 0.5 * (new_harmony[np.random.randint(0, self.dim)] - new_harmony[np.random.randint(0, self.dim)])
                candidate = adjust_value(candidate)
                if func(candidate) < func_value:
                    harmony[i] = candidate  # Update if better
            return harmony

        def global_search(harmony, func):
            new_harmony = np.copy(harmony)
            best_harmony = np.copy(harmony)
            best_fitness = func(best_harmony)

            for i in range(self.dim):
                original_value = new_harmony[i]
                new_harmony[i] = np.random.uniform(max(self.lower_bound, original_value - self.pitch_range),
                                                   min(self.upper_bound, original_value + self.pitch_range))
                new_fitness = func(new_harmony)

                if new_fitness < best_fitness:
                    best_harmony = np.copy(new_harmony)
                    best_fitness = new_fitness

                new_harmony[i] = original_value

            return best_harmony

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

                harmony_memory = local_search(harmony_memory, new_fitness)  # Updated local search

                global_best_harmony = global_search(new_harmony, func)

                for i in range(self.budget):
                    harmony_memory[i] = opposition_based_learning(harmony_memory[i])

                new_harmony_opposite = opposition_based_learning(harmony_memory[np.argmin([func(h) for h in harmony_memory])])
                new_fitness_opposite = func(new_harmony_opposite)

                if new_fitness_opposite < best_fitness:
                    best_solution = np.copy(new_harmony_opposite)
                    best_fitness = new_fitness_opposite
                    pitch = adjust_pitch(pitch, 1)

                harmony_memory[np.argmax([func(h) for h in harmony_memory])] = new_harmony

            return best_solution

        return harmony_search()