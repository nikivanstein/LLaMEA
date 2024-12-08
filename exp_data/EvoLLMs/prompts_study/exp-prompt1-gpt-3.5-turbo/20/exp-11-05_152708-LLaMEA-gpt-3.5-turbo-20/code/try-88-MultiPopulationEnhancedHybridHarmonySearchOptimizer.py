import numpy as np
from scipy.optimize import minimize

class MultiPopulationEnhancedHybridHarmonySearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def generate_harmony():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def improvise(harmony_memory, harmony_memory_size, pitch_adjust_rate, mutation_factor):
            new_harmony = np.copy(harmony_memory[np.random.randint(harmony_memory_size)])
            for i in range(self.dim):
                if np.random.rand() < pitch_adjust_rate:
                    new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
                if np.random.rand() < mutation_factor:
                    new_harmony[i] = self.lower_bound + self.upper_bound - new_harmony[i]  # Opposition-based learning
            return new_harmony

        def differential_evolution(harmony_memory, fitness_values):
            mutation_factor = np.random.uniform(0.1, 0.9)  # Self-adaptive mutation factor
            crossover_rate = 0.9
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

        population_size = 5
        pitch_adjust_rate = 0.1
        harmonies = [np.array([generate_harmony() for _ in range(population_size)]) for _ in range(5)]
        fitness_values = [np.array([func(harmony) for harmony in pop]) for pop in harmonies]

        for _ in range(self.budget - 5*population_size):
            for pop_idx in range(len(harmonies)):
                harmony_memory = harmonies[pop_idx]
                fitness_vals = fitness_values[pop_idx]

                new_harmony = improvise(harmony_memory, population_size, pitch_adjust_rate, 0.5)
                new_fitness = func(new_harmony)
                if new_fitness < np.max(fitness_vals):
                    index = np.argmax(fitness_vals)
                    harmony_memory[index] = new_harmony
                    fitness_vals[index] = new_fitness
                differential_evolution(harmony_memory, fitness_vals)

            best_harmonies = [harmonies[i][np.argmin(fitness_values[i])] for i in range(len(harmonies))]
            best_fitnesses = [np.min(fitness_values[i]) for i in range(len(fitness_values))]

            for i in range(len(harmonies)):
                for j in range(len(harmonies[i])):
                    harmonies[i][j] = best_harmonies[np.random.randint(len(best_harmonies))]

            pitch_adjust_rate = max(0.01, min(0.5, pitch_adjust_rate + 0.1 * (np.sum(fitness_values) - np.sum(best_fitnesses)))

        best_index = np.unravel_index(np.argmin(fitness_values, axis=None), np.shape(fitness_values))
        return harmonies[best_index[0]][best_index[1]]