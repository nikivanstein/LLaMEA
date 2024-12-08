import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # typically 5-10 times dimensionality
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.5  # initial mutation factor
        self.crossover_rate = 0.7  # initial crossover rate
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)

    def __call__(self, func):
        evaluations = 0
        best_idx = None

        # Initialize fitness values for the initial population
        for i in range(self.pop_size):
            if evaluations >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            evaluations += 1

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                # Select three random individuals from the population, different from i
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Mutation with a self-adaptive mutation factor
                F = self.mutation_factor + np.random.rand() * (1.0 - self.mutation_factor)
                mutant_vector = self.population[a] + F * (self.population[b] - self.population[c])
                mutant_vector = np.clip(mutant_vector, self.bounds[0], self.bounds[1])

                # Crossover with a self-adaptive crossover rate
                random_index = np.random.randint(self.dim)
                trial_vector = np.array([mutant_vector[j] if np.random.rand() < self.crossover_rate or j == random_index else self.population[i][j] for j in range(self.dim)])

                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                evaluations += 1

                # Selection
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

            # Update the best solution found
            best_idx = np.argmin(self.fitness)

        return self.population[best_idx], self.fitness[best_idx]