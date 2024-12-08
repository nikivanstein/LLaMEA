import numpy as np
from scipy.optimize import minimize

class RefinedDynamicPitchHybridHarmonySearchOptimizer:
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

        def chaos_optimization(harmony_memory, fitness_values):
            chaos_factor = 0.1
            for i in range(len(harmony_memory)):
                new_harmony = harmony_memory[i] + chaos_factor * np.random.normal(0, 1, self.dim)
                new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
                new_fitness = func(new_harmony)
                if new_fitness < fitness_values[i]:
                    harmony_memory[i] = new_harmony
                    fitness_values[i] = new_fitness

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
            chaos_optimization(harmony_memory, fitness_values)

            local_search_harmony = minimize(func, new_harmony, method='Nelder-Mead').x
            local_search_fitness = func(local_search_harmony)
            if local_search_fitness < np.max(fitness_values):
                index = np.argmax(fitness_values)
                harmony_memory[index] = local_search_harmony
                fitness_values[index] = local_search_fitness

            best_fitness = min(fitness_values)
            pitch_adjust_rate = max(0.01, min(0.5, pitch_adjust_rate + 0.1 * (fitness_values.sum() - best_fitness * len(fitness_values))))

        best_index = np.argmin(fitness_values)
        return harmony_memory[best_index]