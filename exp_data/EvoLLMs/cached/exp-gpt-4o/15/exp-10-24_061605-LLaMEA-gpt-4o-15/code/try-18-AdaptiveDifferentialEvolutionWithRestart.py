import numpy as np

class AdaptiveDifferentialEvolutionWithRestart:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 * dim  # reduced population size for budget efficiency
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8  # slightly reduced mutation factor
        self.crossover_rate = 0.85  # adjusted crossover rate for diversity
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.evaluations = 0

    def __call__(self, func):
        self.initialize_fitness(func)
        
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break

                # Select three random individuals from the population, different from i
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Mutation with adaptive mutation factor
                F = self.mutation_factor * np.random.uniform(0.4, 1.0)  # adaptive scaling
                mutant_vector = self.population[a] + F * (self.population[b] - self.population[c])
                mutant_vector = np.clip(mutant_vector, self.bounds[0], self.bounds[1])

                # Crossover with adaptive strategy
                random_index = np.random.randint(self.dim)
                trial_vector = np.array([mutant_vector[j] if np.random.rand() < self.crossover_rate or j == random_index else self.population[i][j] for j in range(self.dim)])

                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                self.evaluations += 1

                # Selection based on fitness evaluation
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

                # Periodic restart for diversity
                if self.evaluations % (self.budget // 5) == 0:  # restart every 20% of the budget
                    self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
                    self.initialize_fitness(func)

            # Update the best solution found
            best_idx = np.argmin(self.fitness)

        return self.population[best_idx], self.fitness[best_idx]

    def initialize_fitness(self, func):
        for i in range(self.pop_size):
            if self.evaluations >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1