import numpy as np

class AdaptiveDEPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20 + int(3.0 * np.sqrt(self.dim))
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.global_best = None
        self.best_cost = float('inf')

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            for i in range(self.population_size):
                # Select three random indices different from i
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Perform mutation (differential vector)
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                # Perform crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Evaluate the trial solution
                trial_cost = func(trial)
                evals += 1

                # Selection
                if trial_cost < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_cost

                    # Update global best
                    if trial_cost < self.best_cost:
                        self.global_best = trial
                        self.best_cost = trial_cost

                if evals >= self.budget:
                    break

        return self.global_best