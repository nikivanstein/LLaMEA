import numpy as np

class GradientInspiredAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.8  # Differential weight
        self.CR = 0.85  # Initial crossover probability

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Select three distinct indices
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]

                # Mutation with adaptive scaling factor
                F_dyn = 0.6 + 0.4 * (fitness.min() / (fitness[i] + 1e-8))
                mutant = x1 + F_dyn * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Dynamic adjustment of crossover probability
                crossover_prob_adjust = 0.1 * (fitness.min() / (fitness[i] + 1e-8))
                crossover_prob = np.clip(self.CR + crossover_prob_adjust, 0.7, 0.95)
                trial = np.where(np.random.rand(self.dim) < crossover_prob, mutant, population[i])

                # Calculate trial fitness
                trial_fitness = func(trial)
                eval_count += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Adaptive step inspired by quasi-gradient approximation
                gradient_approx = (trial - population[i]) / (np.linalg.norm(trial - population[i]) + 1e-8)
                population[i] += 0.01 * gradient_approx * (fitness.min() - fitness.max())

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]