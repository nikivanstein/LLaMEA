import numpy as np
from scipy.optimize import minimize
import pyswarms as ps

class EnhancedDynamicPitchHybridHarmonySearchOptimizer:
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

        def differential_evolution(harmony_memory, fitness_values):
            mutation_factor = 0.5
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

            # Integrate Particle Swarm Optimization for exploration
            optimizer = ps.single.GlobalBestPSO(n_particles=len(harmony_memory), dimensions=self.dim, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9})
            pso_best = optimizer.optimize(func, iters=10)
            if pso_best[1] < np.max(fitness_values):
                index = np.argmax(fitness_values)
                harmony_memory[index] = pso_best[0]
                fitness_values[index] = pso_best[1]

            best_fitness = min(fitness_values)
            pitch_adjust_rate = max(0.01, min(0.5, pitch_adjust_rate + 0.1 * (fitness_values.sum() - best_fitness * len(fitness_values))))

        best_index = np.argmin(fitness_values)
        return harmony_memory[best_index]