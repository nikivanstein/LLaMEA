import numpy as np

class DynamicAdaptiveDifferentialEvolutionWithNeighborhoodLearning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = max(5, int(budget / (10 * dim)))
        self.f_scale = 0.5  # initial scaling factor for differential evolution
        self.cr_rate = 0.9  # crossover rate
        self.neighborhood_size = max(2, self.pop_size // 5)  # for neighborhood learning

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        best_idx = np.argmin(fitness)
        global_best_position = population[best_idx]
        global_best_fitness = fitness[best_idx]

        while num_evaluations < self.budget:
            for i in range(self.pop_size):
                if num_evaluations >= self.budget:
                    break

                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Mutation
                mutant_vector = population[a] + self.f_scale * (population[b] - population[c])
                mutant_vector = np.clip(mutant_vector, self.lb, self.ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.cr_rate
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])

                # Evaluation
                trial_fitness = func(trial_vector)
                num_evaluations += 1

                # Selection and neighborhood learning
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    if trial_fitness < global_best_fitness:
                        global_best_position = trial_vector
                        global_best_fitness = trial_fitness

                # Neighborhood learning
                neighborhood_indices = np.random.choice(self.pop_size, self.neighborhood_size, replace=False)
                for n_idx in neighborhood_indices:
                    if fitness[n_idx] > fitness[i]:
                        population[n_idx] = population[i]
                        fitness[n_idx] = fitness[i]

        return global_best_position, global_best_fitness