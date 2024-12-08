import numpy as np

class EnhancedAdaptiveDEPS:
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
        self.dynamic_factor = 0.1  # Introduce a factor for population resizing

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        # Adaptive population resizing
        while evals < self.budget:
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                
                # Hybrid exploration: introduce a slight random perturbation
                if np.random.rand() < self.dynamic_factor:
                    trial += np.random.normal(0, 0.1, self.dim)

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

            # Dynamically adjust population size during the run
            if evals < self.budget * 0.5:  # Reduce population size in early stages
                self.population_size = max(5, int(self.population_size * (1.0 - self.dynamic_factor)))
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]

        return self.global_best