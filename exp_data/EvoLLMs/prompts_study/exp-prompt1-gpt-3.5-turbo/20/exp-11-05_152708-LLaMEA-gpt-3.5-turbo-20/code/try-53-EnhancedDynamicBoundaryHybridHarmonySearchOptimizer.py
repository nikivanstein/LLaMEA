import numpy as np
from scipy.optimize import minimize

class EnhancedDynamicBoundaryHybridHarmonySearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def generate_harmony():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def improvise(harmony_memory, harmony_memory_size, pitch_adjust_rate, mutation_factor, crossover_rate):
            new_harmony = np.copy(harmony_memory[np.random.randint(harmony_memory_size)])
            for i in range(self.dim):
                if np.random.rand() < pitch_adjust_rate:
                    new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
            return new_harmony

        def differential_evolution(harmony_memory, fitness_values, mutation_factor, crossover_rate):
            for i in range(len(harmony_memory)):
                target_idx = np.random.choice(list(set(range(len(harmony_memory))) - {i}))
                base, target = harmony_memory[i], harmony_memory[target_idx]
                donor = base + mutation_factor * (target - harmony_memory[np.random.choice(range(len(harmony_memory)))])

                trial = np.copy(base)
                for j in range(len(trial)):
                    if np.random.rand() < crossover_rate:
                        trial[j] = donor[j] if np.random.rand() < 0.5 else base[j]

                trial_fitness = func(trial)
                if trial_fitness < fitness_values[i]:
                    harmony_memory[i] = trial
                    fitness_values[i] = trial_fitness

        harmony_memory_size = 10
        pitch_adjust_rate = 0.1
        mutation_factor = 0.5
        crossover_rate = 0.9
        harmony_memory = np.array([generate_harmony() for _ in range(harmony_memory_size)])
        fitness_values = np.array([func(harmony) for harmony in harmony_memory])

        for _ in range(self.budget - harmony_memory_size):
            new_harmony = improvise(harmony_memory, harmony_memory_size, pitch_adjust_rate, mutation_factor, crossover_rate)
            new_fitness = func(new_harmony)
            if new_fitness < np.max(fitness_values):
                index = np.argmax(fitness_values)
                harmony_memory[index] = new_harmony
                fitness_values[index] = new_fitness
            differential_evolution(harmony_memory, fitness_values, mutation_factor, crossover_rate)

            chaotic_harmony = harmony_memory[np.random.randint(harmony_memory_size)] + 0.1 * np.sin(np.random.standard_normal(self.dim))
            chaotic_harmony = np.clip(chaotic_harmony, self.lower_bound, self.upper_bound)  
            chaotic_fitness = func(chaotic_harmony)
            if chaotic_fitness < np.max(fitness_values):
                index = np.argmax(fitness_values)
                harmony_memory[index] = chaotic_harmony
                fitness_values[index] = chaotic_fitness

            local_search_harmony = minimize(func, harmony_memory[np.argmin(fitness_values)], method='Nelder-Mead').x
            local_search_fitness = func(local_search_harmony)
            if local_search_fitness < np.max(fitness_values):
                index = np.argmax(fitness_values)
                harmony_memory[index] = local_search_harmony
                fitness_values[index] = local_search_fitness

            best_fitness = min(fitness_values)
            pitch_adjust_rate = max(0.01, min(0.5, pitch_adjust_rate + 0.1 * (fitness_values.sum() - best_fitness * len(fitness_values))))
            mutation_factor = max(0.1, min(0.9, mutation_factor + 0.1 * (best_fitness - fitness_values.mean())))
            crossover_rate = max(0.1, min(0.9, crossover_rate + 0.1 * (best_fitness - fitness_values.mean())))

        best_index = np.argmin(fitness_values)
        return harmony_memory[best_index]