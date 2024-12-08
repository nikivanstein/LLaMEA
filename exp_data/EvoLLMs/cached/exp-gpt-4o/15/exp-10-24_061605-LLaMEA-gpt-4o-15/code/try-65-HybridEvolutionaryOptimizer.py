import numpy as np

class HybridEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 * dim
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8  # slightly reduced mutation factor for balance
        self.crossover_rate = 0.8  # slightly reduced crossover rate
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.elite_fraction = 0.2  # preserve a fraction of the best individuals

    def __call__(self, func):
        evaluations = 0

        # Initialize fitness values for the initial population
        for i in range(self.pop_size):
            if evaluations >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            evaluations += 1

        while evaluations < self.budget:
            elite_count = int(self.elite_fraction * self.pop_size)
            elite_indices = np.argsort(self.fitness)[:elite_count]

            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                # Mutation and crossover influenced by elite individuals
                if i in elite_indices:
                    continue

                a, b, c = np.random.choice(elite_indices, 3, replace=True)

                F = self.mutation_factor * np.random.uniform(0.6, 1.0)  # adaptive scaling
                mutant_vector = self.population[a] + F * (self.population[b] - self.population[c])
                mutant_vector = np.clip(mutant_vector, self.bounds[0], self.bounds[1])

                random_index = np.random.randint(self.dim)
                trial_vector = np.array([mutant_vector[j] if np.random.rand() < self.crossover_rate or j == random_index else self.population[i][j] for j in range(self.dim)])

                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                evaluations += 1

                # Selection based on fitness evaluation
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

                # Stochastic perturbation
                if np.random.rand() < 0.2:
                    perturb_vector = self.population[i] + np.random.normal(0, 0.05, self.dim)
                    perturb_vector = np.clip(perturb_vector, self.bounds[0], self.bounds[1])
                    perturb_fitness = func(perturb_vector)
                    evaluations += 1
                    if perturb_fitness < self.fitness[i]:
                        self.population[i] = perturb_vector
                        self.fitness[i] = perturb_fitness

            # Update the best solution found
            best_idx = np.argmin(self.fitness)

        return self.population[best_idx], self.fitness[best_idx]