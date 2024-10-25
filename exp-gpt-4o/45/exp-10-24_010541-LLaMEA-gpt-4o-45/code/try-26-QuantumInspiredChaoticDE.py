import numpy as np

class QuantumInspiredChaoticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.CR = np.random.uniform(0.5, 1.0, self.pop_size)
        self.F = np.random.uniform(0.4, 0.9, self.pop_size)
        self.local_intensification = 0.25  # increased probability for local search
        self.dynamic_scale = 0.3  # increased dynamic scale
        self.chaos_coefficient = 0.6  # slightly adjusted chaos coefficient
    
    def chaotic_map(self, x):
        return self.chaos_coefficient * np.sin(np.pi * x)
    
    def quantum_perturbation(self, x):
        return x + np.random.normal(0, 0.1, size=x.shape)
    
    def __call__(self, func):
        evaluations = 0
        chaos_value = np.random.rand()
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                # Mutation with dynamic scaling and novel strategy
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                dynamic_factor = 1.0 + self.dynamic_scale * (np.random.rand() - 0.5)
                mutant = self.population[a] + dynamic_factor * self.F[i] * (self.population[b] - self.population[c])
                
                # Chaotic Local Search Intensification
                if np.random.rand() < self.local_intensification:
                    local_best = self.population[np.argmin(self.fitness)]
                    mutant = (0.5 + chaos_value) * mutant + (0.5 - chaos_value) * local_best
                
                # Quantum-inspired perturbation
                mutant = self.quantum_perturbation(mutant)
                mutant = np.clip(mutant, *self.bounds)

                # Crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where((np.random.rand(self.dim) < self.CR[i]) | (np.arange(self.dim) == j_rand), mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.CR[i] = 0.85 * self.CR[i] + 0.15 * np.random.rand()  # adjusted adaptive adjustment
                    self.F[i] = 0.85 * self.F[i] + 0.15 * np.random.rand()
                else:
                    self.CR[i] = 0.35 * self.CR[i] + 0.65 * np.random.rand()
                    self.F[i] = 0.35 * self.F[i] + 0.65 * np.random.rand()

                chaos_value = self.chaotic_map(chaos_value)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]