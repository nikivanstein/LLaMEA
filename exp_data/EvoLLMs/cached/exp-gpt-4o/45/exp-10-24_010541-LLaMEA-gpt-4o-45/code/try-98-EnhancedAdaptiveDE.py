import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Increased population size for better search space coverage
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.CR = np.random.uniform(0.5, 1.0, self.pop_size)  # Adjusted CR range for enhanced crossover
        self.F = np.random.uniform(0.4, 0.9, self.pop_size)  # Expanded F range for increased explorative potential
        self.local_intensification = 0.45  # Increased local search probability
        self.dynamic_scale = 0.35  # Increased dynamic scale for diversity
        self.chaos_coefficient = 0.9  # Enhanced chaos for greater exploration
        self.learning_rate = 0.25  # Adjusted adaptation speed
        self.memory = np.zeros(self.dim)  # Memory vector for adaptive learning

    def chaotic_map(self, x):
        return self.chaos_coefficient * x * (1 - x)

    def quasi_opposition(self, x):
        return self.bounds[0] + self.bounds[1] - x

    def __call__(self, func):
        evaluations = 0
        chaos_value = np.random.rand()
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                # Mutation with enhanced strategy
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                dynamic_factor = 1.0 + self.dynamic_scale * (np.random.rand() - 0.5)
                mutant = self.population[a] + dynamic_factor * self.F[i] * (self.population[b] - self.population[c])

                # Chaotic Local Search with Enhanced Memory
                if np.random.rand() < self.local_intensification:
                    local_best = self.population[np.argmin(self.fitness)]
                    mutant = (0.4 + chaos_value) * mutant + (0.6 - chaos_value) * (local_best + self.memory)
                
                # Quasi-oppositional learning
                if np.random.rand() < 0.5:
                    mutant = self.quasi_opposition(mutant)

                mutant = np.clip(mutant, *self.bounds)

                # Crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where((np.random.rand(self.dim) < self.CR[i]) | (np.arange(self.dim) == j_rand), mutant, self.population[i])

                # Selection and Memory Update
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.memory = 0.7 * self.memory + 0.3 * (trial - self.population[i])  # Updated memory weight
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