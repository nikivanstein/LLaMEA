import numpy as np

class HybridEvolutionaryStrategy:
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
        F = 0.7
        CR = 0.85

        while evals < self.budget:
            # Dynamic adjustment based on evaluation progress
            self.population_size = max(5, self.init_population_size - int(evals / self.budget * (self.init_population_size - 5)))

            for i in range(self.population_size):
                # Ensuring diversity through improved selection
                indices = np.random.permutation(self.population_size)
                indices = indices[indices != i]
                a, b, c = population[indices[:3]]

                # Mutation with scaling factor
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                # Crossover with a modified probability
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Evaluation of the trial solution
                trial_cost = func(trial)
                evals += 1

                # Selection process with fitness check
                if trial_cost < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_cost

                    # Update the global best solution
                    if trial_cost < self.best_cost:
                        self.global_best = trial
                        self.best_cost = trial_cost

                # Stop if budget is exhausted
                if evals >= self.budget:
                    break

            # Adaptive F and CR to enhance exploration
            F = 0.6 + 0.2 * np.random.rand()
            CR = 0.75 + 0.15 * np.random.rand()

        return self.global_best