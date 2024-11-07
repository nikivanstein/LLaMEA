import numpy as np
from scipy.optimize import minimize

class DynamicPopulationSizeHybridHarmonySearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def generate_harmony():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def improvise(harmony_memory, pitch_adjust_rate, mutation_factor):
            new_harmony = np.copy(harmony_memory[np.random.randint(len(harmony_memory))])
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

        harmony_memory_size = 10
        population_update_rate = 0.1
        pitch_adjust_rate = 0.1
        harmony_memory = np.array([generate_harmony() for _ in range(harmony_memory_size)])
        fitness_values = np.array([func(harmony) for harmony in harmony_memory])

        for _ in range(self.budget - harmony_memory_size):
            new_harmony = improvise(harmony_memory, pitch_adjust_rate, 0.5)
            new_fitness = func(new_harmony)
            if new_fitness < np.max(fitness_values):
                index = np.argmax(fitness_values)
                harmony_memory[index] = new_harmony
                fitness_values[index] = new_fitness
            differential_evolution(harmony_memory, fitness_values)

            if np.random.rand() < population_update_rate:
                harmony_memory = np.vstack([harmony_memory, generate_harmony()])
                fitness_values = np.append(fitness_values, func(harmony_memory[-1]))

            best_index = np.argmin(fitness_values)
            best_fitness = fitness_values[best_index]
            pitch_adjust_rate = max(0.01, min(0.5, pitch_adjust_rate + 0.1 * (fitness_values.sum() - best_fitness * len(fitness_values))))

        return harmony_memory[best_index]