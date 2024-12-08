import numpy as np

class OptimizedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.7
        self.CR = 0.85  # Initial crossover rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def opposition_based_learning(self, individual):
        return self.lower_bound + self.upper_bound - individual

    def local_search(self, x, func):
        trial = x + np.random.uniform(-0.05, 0.05, self.dim)
        trial = np.clip(trial, self.lower_bound, self.upper_bound)
        return trial if func(trial) < func(x) else x

    def mutate(self, population, idx):
        indices = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = indices
        while a == idx:
            a = np.random.choice(self.population_size)
        diversity_factor = np.exp(-np.std(population, axis=0).mean())
        mutant = population[a] + self.F * diversity_factor * (population[b] - population[c])
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant, current_iteration, max_iterations):
        dynamic_CR = self.CR - (0.5 * current_iteration / max_iterations)  # Dynamic crossover adjustment
        cross_points = np.random.rand(self.dim) < dynamic_CR
        trial = np.where(cross_points, mutant, target)
        return trial

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size
        max_iterations = self.budget // self.population_size

        for current_iteration in range(max_iterations):
            if budget_used >= self.budget:
                break
            for i in range(self.population_size):
                mutant = self.mutate(population, i)
                trial = self.crossover(population[i], mutant, current_iteration, max_iterations)
                trial = self.local_search(trial, func)

                trial_fitness = func(trial)
                budget_used += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used < self.budget and np.random.rand() < 0.5:
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