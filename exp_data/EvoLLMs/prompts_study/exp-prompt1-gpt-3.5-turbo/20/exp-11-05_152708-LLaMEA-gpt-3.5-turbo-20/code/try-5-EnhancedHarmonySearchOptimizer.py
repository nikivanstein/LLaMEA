import numpy as np

class EnhancedHarmonySearchOptimizer(HarmonySearchOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.crossover_rate = 0.9
        self.mutation_factor = 0.5

    def __call__(self, func):
        def DE_crossover(population, target_idx):
            r1, r2, r3 = np.random.choice(len(population), 3, replace=False)
            mutant = population[r1] + self.mutation_factor * (population[r2] - population[r3])
            crossover_mask = np.random.rand(self.dim) < self.crossover_rate
            trial = np.where(crossover_mask, mutant, population[target_idx])
            return trial

        harmony_memory_size = 10
        pitch_adjust_rate = 0.1
        harmony_memory = np.array([self.generate_harmony() for _ in range(harmony_memory_size)])
        fitness_values = np.array([func(harmony) for harmony in harmony_memory])

        for _ in range(self.budget - harmony_memory_size):
            new_harmony = self.improvise(harmony_memory, harmony_memory_size, pitch_adjust_rate)
            trial = DE_crossover(harmony_memory, np.random.randint(harmony_memory_size))
            new_fitness = func(trial)
            if new_fitness < np.max(fitness_values):
                index = np.argmax(fitness_values)
                harmony_memory[index] = trial
                fitness_values[index] = new_fitness

        best_index = np.argmin(fitness_values)
        return harmony_memory[best_index]