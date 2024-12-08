import numpy as np

class CompactDynamicDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 * dim  # reduced population size for budget efficiency
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8  # adjusted mutation factor for better control
        self.crossover_rate = 0.85  # adjusted crossover rate for effective recombination
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)

    def __call__(self, func):
        evaluations = 0

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

                # Dynamic leadership: Use the best individual found so far for mutation
                best_idx = np.argmin(self.fitness)
                F = self.mutation_factor * np.random.uniform(0.6, 1.0)  # adaptive mutation
                mutant_vector = self.population[best_idx] + F * (self.population[b] - self.population[c])
                mutant_vector = np.clip(mutant_vector, self.bounds[0], self.bounds[1])

                # Crossover with adaptive strategy
                random_index = np.random.randint(self.dim)
                trial_vector = np.array([mutant_vector[j] if np.random.rand() < self.crossover_rate or j == random_index else self.population[i][j] for j in range(self.dim)])

                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                evaluations += 1

                # Selection based on fitness evaluation
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

            # Stochastic local search with reduced probability
            if np.random.rand() < 0.08:  # 8% chance for local search
                local_vector = trial_vector + np.random.normal(0, 0.1, self.dim)
                local_vector = np.clip(local_vector, self.bounds[0], self.bounds[1])
                local_fitness = func(local_vector)
                evaluations += 1
                if local_fitness < trial_fitness:
                    self.population[i] = local_vector
                    self.fitness[i] = local_fitness

        return self.population[best_idx], self.fitness[best_idx]