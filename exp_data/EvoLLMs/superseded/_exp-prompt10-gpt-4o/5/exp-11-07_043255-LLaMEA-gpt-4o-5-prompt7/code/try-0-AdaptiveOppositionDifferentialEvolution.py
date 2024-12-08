import numpy as np

class AdaptiveOppositionDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def opposition_based_learning(self, population):
        return self.lower_bound + self.upper_bound - population

    def local_search(self, x, func):
        for i in range(5):  # Local search steps
            trial = x + np.random.uniform(-0.1, 0.1, self.dim)
            trial = np.clip(trial, self.lower_bound, self.upper_bound)
            if func(trial) < func(x):
                x = trial
        return x

    def mutate(self, population, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = population[a] + self.F * (population[b] - population[c])
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
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

                # Opposition-based learning
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

# Example usage:
# optimizer = AdaptiveOppositionDifferentialEvolution(budget=1000, dim=2)
# best_solution = optimizer(some_black_box_function)