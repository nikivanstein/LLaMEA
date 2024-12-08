import numpy as np

class HybridMC_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        np.random.seed(0)
        population_size = 10
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evaluations = population_size

        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Differential Evolution Mutation with adaptive scale
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                diversity = np.std(pop) / np.mean(np.abs(pop))  # Relative diversity
                scale = 0.5 + 0.3 * (diversity / (self.upper_bound - self.lower_bound))
                mutant = pop[a] + scale * (pop[b] - pop[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Monte Carlo exploration with random noise
                trial = np.where(np.random.rand(self.dim) < 0.5, mutant + np.random.normal(0, 0.1, self.dim), pop[i])
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness

        return best