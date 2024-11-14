import numpy as np

class DifferentialEvolutionWithAdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic for population size
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_position = population[best_idx]
        best_fitness = fitness[best_idx]

        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break

                # Select three random indices different from i
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Perform mutation and crossover
                mutant = population[a] + self.mutation_factor * (population[b] - population[c])
                mutant = np.clip(mutant, self.lb, self.ub)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])

                # Evaluate the trial vector
                trial_fitness = func(trial)
                num_evaluations += 1

                # Adapt mutation strategy based on performance
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_position = trial
                        best_fitness = trial_fitness
                else:
                    self.mutation_factor = np.random.uniform(0.5, 1.0)  # adaptively modify mutation factor

        return best_position, best_fitness