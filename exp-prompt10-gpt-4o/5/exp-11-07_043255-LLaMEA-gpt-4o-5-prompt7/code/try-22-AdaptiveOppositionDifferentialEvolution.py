import numpy as np

class AdaptiveOppositionDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.7  # Increased differential weight for enhanced exploration
        self.CR = 0.85  # Adjusted crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def opposition_based_learning(self, population):
        return self.lower_bound + self.upper_bound - population

    def local_search(self, x, func):
        trial = x + np.random.uniform(-0.05, 0.05, self.dim)  # Narrowed exploration range
        trial = np.clip(trial, self.lower_bound, self.upper_bound)
        if func(trial) < func(x):
            x = trial
        return x

    def mutate(self, population, idx):
        a, b, c = np.random.choice(self.population_size, 3, replace=False)
        mutant = population[a] + self.F * (population[b] - population[c])
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        trial = np.where(cross_points, mutant, target)
        return trial

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(population, i)
                trial = self.crossover(population[i], mutant)
                trial = self.local_search(trial, func)

                trial_fitness = func(trial)
                budget_used += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used < self.budget:
                    opposite = self.opposition_based_learning(population[i])
                    opposite_fitness = func(opposite)
                    budget_used += 1
                    if opposite_fitness < fitness[i]:
                        population[i] = opposite
                        fitness[i] = opposite_fitness

                if budget_used >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx]