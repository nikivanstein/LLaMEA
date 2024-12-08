import numpy as np

class AdaptiveQuantumChaoticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 12 * dim  # Increased population size for diverse search
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.CR = np.random.uniform(0.6, 1.0, self.pop_size)  # Enhanced crossover range
        self.F = np.random.uniform(0.3, 0.8, self.pop_size)  # Adjusted mutation factor
        self.local_intensification = 0.4  # Enhanced probability for local search
        self.dynamic_scale = 0.3  # Increased dynamic scale for mutation
        self.chaos_coefficient = 0.85  # Adjusted chaos coefficient
        self.learning_rate = 0.2  # Enhanced learning rate for faster adaptation
        self.memory = np.zeros(self.dim)  # Memory vector for temporal learning
        self.quantum_factor = 0.2  # Added quantum factor for superposition

    def chaotic_map(self, x):
        return self.chaos_coefficient * x * (1 - x)

    def quantum_superposition(self, position):
        return position + self.quantum_factor * np.random.uniform(-1, 1, self.dim)

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
                    mutant = (0.5 + chaos_value) * mutant + (0.5 - chaos_value) * (local_best + self.memory)
                
                mutant = np.clip(mutant, *self.bounds)
                mutant = self.quantum_superposition(mutant)  # Quantum inspired mutation

                j_rand = np.random.randint(self.dim)
                trial = np.where((np.random.rand(self.dim) < self.CR[i]) | (np.arange(self.dim) == j_rand), mutant, self.population[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.memory = 0.5 * self.memory + 0.5 * (trial - self.population[i])
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.CR[i] = (1 - self.learning_rate) * self.CR[i] + self.learning_rate * np.random.rand()
                    self.F[i] = (1 - self.learning_rate) * self.F[i] + self.learning_rate * np.random.rand()
                else:
                    self.CR[i] = (1 - self.learning_rate) * np.random.rand() + self.learning_rate * self.CR[i]
                    self.F[i] = (1 - self.learning_rate) * np.random.rand() + self.learning_rate * self.F[i]

                chaos_value = self.chaotic_map(chaos_value)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]