import numpy as np

class HybridQuantumInspiredChaoticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.CR = np.random.uniform(0.6, 1.0, self.pop_size)  # adjusted crossover rate range
        self.F = np.random.uniform(0.5, 0.8, self.pop_size)  # adjusted scaling factor range
        self.local_intensification = 0.25  # adjusted probability for local search
        self.dynamic_scale = 0.3  # adjusted dynamic scale
        self.chaos_coefficient = 0.85  # increased chaos coefficient

    def chaotic_map(self, x):
        return self.chaos_coefficient * x * (1 - x)
    
    def quantum_grover_mutation(self, individual, best):
        theta = np.arccos(individual / np.linalg.norm(individual))
        new_theta = theta + np.random.uniform(-np.pi, np.pi, size=individual.shape)
        return best + np.linalg.norm(best) * np.cos(new_theta)

    def __call__(self, func):
        evaluations = 0
        chaos_value = np.random.rand()
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                dynamic_factor = 1.0 + self.dynamic_scale * (np.random.rand() - 0.5)
                mutant = self.population[a] + dynamic_factor * self.F[i] * (self.population[b] - self.population[c])
                
                if np.random.rand() < self.local_intensification:
                    local_best = self.population[np.argmin(self.fitness)]
                    mutant = self.quantum_grover_mutation(mutant, local_best)
                
                mutant = np.clip(mutant, *self.bounds)

                j_rand = np.random.randint(self.dim)
                trial = np.where((np.random.rand(self.dim) < self.CR[i]) | (np.arange(self.dim) == j_rand), mutant, self.population[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.CR[i] = 0.7 * self.CR[i] + 0.3 * np.random.rand()  # modified adaptive adjustment
                    self.F[i] = 0.7 * self.F[i] + 0.3 * np.random.rand()
                else:
                    self.CR[i] = 0.5 * self.CR[i] + 0.5 * np.random.rand()
                    self.F[i] = 0.5 * self.F[i] + 0.5 * np.random.rand()

                chaos_value = self.chaotic_map(chaos_value)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]