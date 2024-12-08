import numpy as np
from scipy.optimize import minimize

class EnhancedDynamicBoundaryHybridHarmonySearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def generate_harmony(population_size):
            return np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))

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

        population_sizes = [10, 20]
        pitch_adjust_rates = [0.1, 0.2]
        harmony_memories = [generate_harmony(pop_size) for pop_size in population_sizes]
        fitness_values = [np.array([func(harmony) for harmony in harmony_memory]) for harmony_memory in harmony_memories]

        for _ in range(self.budget - max(population_sizes)):
            for i in range(len(population_sizes)):
                new_harmony = improvise(harmony_memories[i], population_sizes[i], pitch_adjust_rates[i], 0.5)
                new_fitness = func(new_harmony)
                if new_fitness < np.max(fitness_values[i]):
                    index = np.argmax(fitness_values[i])
                    harmony_memories[i][index] = new_harmony
                    fitness_values[i][index] = new_fitness
                differential_evolution(harmony_memories[i], fitness_values[i])

            best_fitnesses = [min(fitness) for fitness in fitness_values]
            pitch_adjust_rates = [max(0.01, min(0.5, rate + 0.1 * (fitness.sum() - best * len(fitness)))
                                   for rate, fitness, best in zip(pitch_adjust_rates, fitness_values, best_fitnesses)]

        best_indices = [np.argmin(fitness) for fitness in fitness_values]
        best_harmonies = [harmony_memory[best_index] for harmony_memory, best_index in zip(harmony_memories, best_indices)]
        return min(best_harmonies, key=func)