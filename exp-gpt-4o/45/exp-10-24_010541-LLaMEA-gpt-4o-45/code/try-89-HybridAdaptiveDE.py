import numpy as np

class HybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Retaining population size for broad search scope
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.CR = np.random.uniform(0.5, 0.9, self.pop_size)  # Fine-tuned CR range for crossover balance
        self.F = np.random.uniform(0.3, 0.8, self.pop_size)  # Adjusted F range for controlled exploration
        self.opposition_threshold = 0.3  # Probability threshold for opposition learning
        self.learning_rate = 0.2  # Adjusted learning rate for parameter self-adaptation
        self.memory = np.zeros(self.dim)

    def chaotic_map(self, x):
        return 0.9 * x * (1 - x)

    def opposition_based_learning(self, x):
        return self.bounds[0] + self.bounds[1] - x

    def __call__(self, func):
        evaluations = 0
        chaos_value = np.random.rand()
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                # Mutation with an adaptive strategy
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.F[i] * (self.population[b] - self.population[c])

                # Opposition-based Local Search
                if np.random.rand() < self.opposition_threshold:
                    opposition = self.opposition_based_learning(mutant)
                    mutant = 0.5 * (mutant + opposition)

                mutant = np.clip(mutant, *self.bounds)

                # Crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where((np.random.rand(self.dim) < self.CR[i]) | (np.arange(self.dim) == j_rand), mutant, self.population[i])

                # Selection and Memory Update
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.memory = 0.6 * self.memory + 0.4 * (trial - self.population[i])
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