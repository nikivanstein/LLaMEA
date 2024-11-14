import numpy as np

class EnhancedDynamicDEPS:
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
        F = 0.8
        CR = 0.9

        while evals < self.budget:
            # Adjust population size dynamically
            self.population_size = self.init_population_size - int(evals / self.budget * (self.init_population_size - 5))

            for i in range(self.population_size):
                # Perform stochastic tournament selection
                idxs = np.random.choice(range(self.population_size), 5, replace=False)
                ranked = sorted(idxs, key=lambda idx: fitness[idx])
                a, b, c = population[ranked[:3]]

                # Perform mutation (differential vector)
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                # Perform crossover
                cross_points = np.random.rand(self.dim) < CR
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

            # Self-adaptive F and CR
            F = 0.5 + 0.5 * np.random.rand()
            CR = 0.7 + 0.2 * np.random.rand()

        return self.global_best