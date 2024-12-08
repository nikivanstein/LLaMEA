import numpy as np

class HybridGeneticDifferentialAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic for population size
        self.F_base = 0.5  # base scaling factor for mutation
        self.CR = 0.9  # crossover probability

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break

                # Adaptive Differential Mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                F = self.F_base + np.random.rand() * 0.5  # Adaptive scaling factor
                mutated_vector = population[a] + F * (population[b] - population[c])
                mutated_vector = np.clip(mutated_vector, self.lb, self.ub)

                # Genetic Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                offspring = np.where(crossover_mask, mutated_vector, population[i])

                # Evaluate offspring
                offspring_fitness = func(offspring)
                num_evaluations += 1

                # Selection
                if offspring_fitness < fitness[i]:
                    population[i] = offspring
                    fitness[i] = offspring_fitness
                    if offspring_fitness < best_fitness:
                        best_individual = offspring
                        best_fitness = offspring_fitness

        return best_individual, best_fitness