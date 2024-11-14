import numpy as np

class EnhancedDynamicPitchHarmonySearch(DynamicPitchHarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = 0.5
        self.crossover_rate = 0.9

    def mutation(self, harmony):
        mutant_harmony = np.clip(harmony + self.mutation_rate * (np.random.uniform(self.lower_bound, self.upper_bound, self.dim) - harmony), self.lower_bound, self.upper_bound)
        return mutant_harmony

    def crossover(self, harmony, other_harmony):
        mask = np.random.choice([True, False], self.dim, p=[self.crossover_rate, 1 - self.crossover_rate])
        new_harmony = np.where(mask, harmony, other_harmony)
        return new_harmony

    def harmony_search(self):
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

            index = np.random.randint(self.dim)
            new_harmony[index] = np.random.uniform(max(self.lower_bound, new_harmony[index] - pitch),
                                                   min(self.upper_bound, new_harmony[index] + pitch))

            new_harmony_opposite = opposition_based_learning(new_harmony)
            new_fitness_opposite = func(new_harmony_opposite)

            if new_fitness_opposite < best_fitness:
                best_solution = np.copy(new_harmony_opposite)
                best_fitness = new_fitness_opposite

            mutant_harmony = self.mutation(new_harmony)
            crossover_harmony = self.crossover(new_harmony, harmony_memory[np.random.randint(self.budget)])

            harmony_memory[np.argmax([func(h) for h in harmony_memory])] = new_harmony
            harmony_memory[np.argmax([func(m) for m in [mutant_harmony, crossover_harmony]])] = mutant_harmony if func(
                mutant_harmony) < func(crossover_harmony) else crossover_harmony

            pitch = adjust_pitch(pitch)

        return best_solution