import numpy as np

class EnhancedAdaptiveDEPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20 + int(3.0 * np.sqrt(self.dim))
        self.global_best = None
        self.best_cost = float('inf')
        self.init_population_size = self.population_size

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size
        F = np.random.uniform(0.5, 0.9)  # Stochastic F
        CR = np.random.uniform(0.8, 1.0)  # Stochastic CR

        while evals < self.budget:
            self.population_size = self.init_population_size - int(evals / self.budget * (self.init_population_size - 5))

            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Enhanced mutation using weighted average of best and random
                mutant = np.clip(a + F * (b - c) + 0.5 * F * (self.global_best - a), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                trial_cost = func(trial)
                evals += 1

                if trial_cost < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_cost

                    if trial_cost < self.best_cost:
                        self.global_best = trial
                        self.best_cost = trial_cost

                if evals >= self.budget:
                    break

            F = np.random.uniform(0.5, 0.9)  # Stochastic F update
            CR = np.random.uniform(0.8, 1.0)  # Stochastic CR update

        return self.global_best