import numpy as np

class EnhancedHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.bandwidth = 0.5

    def __call__(self, func):
        def adjust_value(value, bandwidth):
            return np.clip(value, value - bandwidth, value + bandwidth)

        def enhanced_harmony_search():
            harmony_memory = [initialize_harmony() for _ in range(self.budget)]
            best_solution = np.copy(harmony_memory[0])
            best_fitness = func(best_solution)

            for _ in range(self.budget):
                new_harmony = np.mean(harmony_memory, axis=0)
                new_harmony = adjust_value(new_harmony, self.bandwidth)
                new_fitness = func(new_harmony)

                if new_fitness < best_fitness:
                    best_solution = np.copy(new_harmony)
                    best_fitness = new_fitness

                index = np.random.randint(self.dim)
                new_harmony[index] = np.random.uniform(self.lower_bound, self.upper_bound)

                harmony_memory[np.argmax([func(h) for h in harmony_memory])] = new_harmony

                self.bandwidth *= 0.98  # Dynamic adaptation of bandwidth for better exploration

            return best_solution

        return enhanced_harmony_search()