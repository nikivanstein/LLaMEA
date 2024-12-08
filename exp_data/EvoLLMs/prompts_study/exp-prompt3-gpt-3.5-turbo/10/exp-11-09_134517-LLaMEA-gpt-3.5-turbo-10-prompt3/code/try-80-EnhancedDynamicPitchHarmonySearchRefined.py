import numpy as np

class EnhancedDynamicPitchHarmonySearchRefined(EnhancedDynamicPitchHarmonySearch):
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

        shuffled_harmonies = [np.random.permutation(harmony) for _ in range(self.budget)]
        for shuffled_harmony in shuffled_harmonies:
            new_shuffled_harmony = global_search(shuffled_harmony, func)
            new_shuffled_fitness = func(new_shuffled_harmony)
            
            if new_shuffled_fitness < best_fitness:
                best_harmony = np.copy(new_shuffled_harmony)
                best_fitness = new_shuffled_fitness
                
        return best_harmony