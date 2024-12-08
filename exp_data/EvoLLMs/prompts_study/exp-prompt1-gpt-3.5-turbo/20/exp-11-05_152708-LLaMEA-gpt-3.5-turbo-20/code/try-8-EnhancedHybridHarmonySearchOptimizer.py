import numpy as np

class EnhancedHybridHarmonySearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def generate_harmony():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def improvise(harmony_memory, harmony_memory_size, pitch_adjust_rate):
            new_harmony = np.copy(harmony_memory[np.random.randint(harmony_memory_size)])
            for i in range(self.dim):
                if np.random.rand() < pitch_adjust_rate:
                    new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
            return new_harmony

        def artificial_immune_mutation(harmony_memory):
            mutation_rate = 0.1
            for i in range(len(harmony_memory)):
                for j in range(self.dim):
                    if np.random.rand() < mutation_rate:
                        harmony_memory[i][j] = np.clip(harmony_memory[i][j] + np.random.normal(0, 1), self.lower_bound, self.upper_bound)

        harmony_memory_size = 10
        pitch_adjust_rate = 0.1
        harmony_memory = np.array([generate_harmony() for _ in range(harmony_memory_size)])
        fitness_values = np.array([func(harmony) for harmony in harmony_memory])

        for _ in range(self.budget - harmony_memory_size):
            new_harmony = improvise(harmony_memory, harmony_memory_size, pitch_adjust_rate)
            new_fitness = func(new_harmony)
            if new_fitness < np.max(fitness_values):
                index = np.argmax(fitness_values)
                harmony_memory[index] = new_harmony
                fitness_values[index] = new_fitness
            differential_evolution(harmony_memory, fitness_values)
            artificial_immune_mutation(harmony_memory)

        best_index = np.argmin(fitness_values)
        return harmony_memory[best_index]