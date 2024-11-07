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

        def select_crowded_harmony(harmony_memory, fitness_values, num_select):
            sorted_indices = np.argsort(fitness_values)
            return harmony_memory[sorted_indices[:num_select]], fitness_values[sorted_indices[:num_select]]

        def mutate_harmony(harmony, mutation_rate):
            mutated_harmony = np.copy(harmony)
            for i in range(self.dim):
                if np.random.rand() < mutation_rate:
                    mutated_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
            return mutated_harmony

        harmony_memory_size = 10
        mutation_rate = 0.5
        pitch_adjust_rate = 0.1
        harmony_memory = np.array([generate_harmony() for _ in range(harmony_memory_size)])
        fitness_values = np.array([func(harmony) for harmony in harmony_memory])

        for _ in range(self.budget - harmony_memory_size):
            new_harmony = mutate_harmony(harmony_memory[np.random.randint(harmony_memory_size)], mutation_rate)
            new_fitness = func(new_harmony)
            if new_fitness < np.max(fitness_values):
                index = np.argmax(fitness_values)
                harmony_memory[index] = new_harmony
                fitness_values[index] = new_fitness

            selected_harmonies, selected_fitness = select_crowded_harmony(harmony_memory, fitness_values, harmony_memory_size // 2)

            for i in range(harmony_memory_size):
                base, target = selected_harmonies[i % (harmony_memory_size // 2)], selected_harmonies[i]
                donor = base + np.random.uniform() * (target - selected_harmonies[np.random.choice(harmony_memory_size // 2)])

                trial = np.copy(base)
                for j in range(len(trial)):
                    if np.random.rand() < 0.9:
                        trial[j] = donor[j] if np.random.rand() < 0.5 else base[j]

                trial_fitness = func(trial)
                if trial_fitness < selected_fitness[i % (harmony_memory_size // 2)]:
                    selected_harmonies[i % (harmony_memory_size // 2)] = trial
                    selected_fitness[i % (harmony_memory_size // 2)] = trial_fitness

            harmony_memory[:harmony_memory_size // 2] = selected_harmonies
            fitness_values[:harmony_memory_size // 2] = selected_fitness

            best_index = np.argmin(fitness_values)
            best_harmony = harmony_memory[best_index]

            pitch_adjust_rate = max(0.01, min(0.5, pitch_adjust_rate + 0.1 * (fitness_values.sum() - func(best_harmony) * len(fitness_values))))

        return best_harmony