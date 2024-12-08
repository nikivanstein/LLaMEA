import numpy as np

class EnhancedDynamicPitchHarmonySearch(DynamicPitchHarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.diversity_rate = 0.5

    def __call__(self, func):
        def harmony_search():
            harmony_memory = [initialize_harmony() for _ in range(self.budget)]
            best_solution = np.copy(harmony_memory[0])
            best_fitness = func(best_solution)
            pitch = self.pitch_range

            for _ in range(self.budget):
                new_harmony = np.mean(harmony_memory, axis=0)
                new_harmony = adjust_value(new_harmony)
                new_fitness = func(new_harmony)
                diversity = np.mean([np.linalg.norm(h - new_harmony) for h in harmony_memory])

                if new_fitness < best_fitness:
                    best_solution = np.copy(new_harmony)
                    best_fitness = new_fitness
                    pitch = adjust_pitch(pitch, 1)

                pitch = adjust_pitch(pitch, diversity * self.diversity_rate)

                for i in range(self.budget):
                    harmony_memory[i] = opposition_based_learning(harmony_memory[i])

                index = np.random.randint(self.dim)
                new_harmony[index] = np.random.uniform(max(self.lower_bound, new_harmony[index] - pitch),
                                                       min(self.upper_bound, new_harmony[index] + pitch))

                new_harmony_opposite = opposition_based_learning(new_harmony)
                new_fitness_opposite = func(new_harmony_opposite)

                if new_fitness_opposite < best_fitness:
                    best_solution = np.copy(new_harmony_opposite)
                    best_fitness = new_fitness_opposite
                    pitch = adjust_pitch(pitch, 1)

                harmony_memory[np.argmax([func(h) for h in harmony_memory])] = new_harmony

            return best_solution

        return harmony_search()