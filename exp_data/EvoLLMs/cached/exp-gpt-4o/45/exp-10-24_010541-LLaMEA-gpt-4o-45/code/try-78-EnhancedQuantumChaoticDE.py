import numpy as np

class EnhancedQuantumChaoticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Increased population size for diversity
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.CR = np.random.uniform(0.3, 0.9, self.pop_size)  # Adjusted CR range for exploration
        self.F = np.random.uniform(0.2, 0.7, self.pop_size)  # Adjusted F range for exploration
        self.local_intensification = 0.45  # Adjusted local search probability
        self.dynamic_scale = 0.25  # Reduced dynamic scale for coherent variability
        self.chaos_coefficient = 0.9  # Enhanced chaos for exploratory dynamics
        self.learning_rate = 0.35  # Adjusted adaptation speed
        self.memory = np.zeros(self.dim)  # Memory vector for temporal learning

    def chaotic_map(self, x):
        return self.chaos_coefficient * np.sin(np.pi * x)

    def quantum_perturbation(self, solution):
        # Introduce a quantum perturbation for exploiting new regions
        perturbation = np.random.normal(0, 0.1, self.dim)
        return np.clip(solution + perturbation, *self.bounds)

    def __call__(self, func):
        evaluations = 0
        chaos_value = np.random.rand()
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                # Mutation with dynamic scaling and refined strategy
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                dynamic_factor = 1.0 + self.dynamic_scale * chaos_value
                mutant = self.population[a] + dynamic_factor * self.F[i] * (self.population[b] - self.population[c])

                # Chaotic Local Search with Memory and Quantum Perturbation
                if np.random.rand() < self.local_intensification:
                    local_best = self.population[np.argmin(self.fitness)]
                    mutant = (0.5 + chaos_value) * mutant + (0.5 - chaos_value) * (local_best + self.memory)
                    mutant = self.quantum_perturbation(mutant)
                
                mutant = np.clip(mutant, *self.bounds)

                # Crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where((np.random.rand(self.dim) < self.CR[i]) | (np.arange(self.dim) == j_rand), mutant, self.population[i])

                # Selection and Memory Update
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.memory = 0.5 * self.memory + 0.5 * (trial - self.population[i])  # Memory weight balanced
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