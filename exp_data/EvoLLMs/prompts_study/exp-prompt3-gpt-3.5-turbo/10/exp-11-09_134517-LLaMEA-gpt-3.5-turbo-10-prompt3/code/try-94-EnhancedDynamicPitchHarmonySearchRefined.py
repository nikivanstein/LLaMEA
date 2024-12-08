import numpy as np

class EnhancedDynamicPitchHarmonySearchRefined(EnhancedDynamicPitchHarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.opposition_rate = 0.1

    def __call__(self, func):
        def adjust_value(value):
            return np.clip(value, self.lower_bound, self.upper_bound)

        def opposition_based_learning(value):
            return self.lower_bound + self.upper_bound - value

        def refine_local_search(harmony, func_value):
            new_harmony = np.copy(harmony)
            for i in range(self.dim):
                original_value = new_harmony[i]
                new_value = np.random.uniform(max(self.lower_bound, original_value - self.pitch_range),
                                              min(self.upper_bound, original_value + self.pitch_range))
                new_value = adjust_value(new_value)
                new_fitness = func(new_value)
                if new_fitness < func_value:
                    harmony[i] = new_value
            return new_harmony

        def enhance_global_search(harmony, func):
            new_harmony = np.copy(harmony)
            best_harmony = np.copy(harmony)
            for i in range(self.dim):
                original_value = new_harmony[i]
                new_value = np.random.uniform(max(self.lower_bound, original_value - self.pitch_range),
                                              min(self.upper_bound, original_value + self.pitch_range))
                new_value = adjust_value(new_value)
                new_fitness = func(new_value)
                if new_fitness < func(best_harmony):
                    best_harmony[i] = new_value
            return best_harmony

        def harmony_search_refined():
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
                    self.pitch_range *= np.exp(self.pitch_adapt_rate)

                for i in range(self.dim):
                    harmony_memory[i] = opposition_based_learning(harmony_memory[i])

                local_best_harmony = refine_local_search(new_harmony, new_fitness)
                global_best_harmony = enhance_global_search(new_harmony, func)

                for i in range(self.budget):
                    harmony_memory[i] = opposition_based_learning(harmony_memory[i])

                new_harmony_opposite = opposition_based_learning(local_best_harmony)
                new_fitness_opposite = func(new_harmony_opposite)

                if new_fitness_opposite < best_fitness:
                    best_solution = np.copy(new_harmony_opposite)
                    best_fitness = new_fitness_opposite
                    self.pitch_range *= np.exp(self.pitch_adapt_rate)

                harmony_memory[np.argmax([func(h) for h in harmony_memory])] = new_harmony

            return best_solution

        return harmony_search_refined()