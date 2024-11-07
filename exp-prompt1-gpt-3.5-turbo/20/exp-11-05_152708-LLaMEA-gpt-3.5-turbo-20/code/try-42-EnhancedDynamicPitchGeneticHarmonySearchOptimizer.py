import numpy as np
from scipy.optimize import minimize

class EnhancedDynamicPitchGeneticHarmonySearchOptimizer:
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

        def genetic_search(harmony_memory, fitness_values):
            pop_size = len(harmony_memory)
            crossover_rate = 0.7
            mutation_rate = 0.1
            elite_size = 2

            # Selection
            elite_indices = np.argsort(fitness_values)[:elite_size]
            elite_population = harmony_memory[elite_indices]

            # Crossover
            for _ in range(pop_size - elite_size):
                idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
                parent1, parent2 = harmony_memory[idx1], harmony_memory[idx2]
                child = np.where(np.random.rand(self.dim) < crossover_rate, parent1, parent2)
                harmony_memory[idx1] = child

            # Mutation
            for i in range(pop_size):
                if i not in elite_indices and np.random.rand() < mutation_rate:
                    harmony_memory[i] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

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
            genetic_search(harmony_memory, fitness_values)

            # Local search using Nelder-Mead method
            local_search_harmony = minimize(func, harmony_memory[np.argmin(fitness_values)], method='Nelder-Mead').x
            local_search_fitness = func(local_search_harmony)
            if local_search_fitness < np.max(fitness_values):
                index = np.argmax(fitness_values)
                harmony_memory[index] = local_search_harmony
                fitness_values[index] = local_search_fitness

            best_fitness = min(fitness_values)
            pitch_adjust_rate = max(0.01, min(0.5, pitch_adjust_rate + 0.1 * (fitness_values.sum() - best_fitness * len(fitness_values))))

        best_index = np.argmin(fitness_values)
        return harmony_memory[best_index]