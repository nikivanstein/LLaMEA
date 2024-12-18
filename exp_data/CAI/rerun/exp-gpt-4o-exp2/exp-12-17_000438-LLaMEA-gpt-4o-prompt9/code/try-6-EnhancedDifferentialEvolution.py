import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * self.dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.CR = 0.9  # Crossover rate
        self.F = 0.8   # Differential weight
        self.mutation_factor_bounds = (0.5, 1.0)
        self.dynamic_population_factor = 0.5

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            # Dynamically adjust population size
            adaptive_population_size = int(self.population_size * (1 - self.dynamic_population_factor * evaluations / self.budget))
            new_population = np.zeros_like(self.population[:adaptive_population_size])

            for i in range(adaptive_population_size):
                # Self-adaptive mutation strategy
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                F_dynamic = self.F + (np.random.rand() - 0.5) * 0.2  # Adjusted scaling factor range
                mutant = np.clip(a + F_dynamic * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial
                if trial_fitness < func(self.population[i]):
                    new_population[i] = trial
                else:
                    new_population[i] = self.population[i]

                if evaluations >= self.budget:
                    break

            if evaluations >= self.budget:
                break

            # Update population with new solutions
            self.population[:adaptive_population_size] = new_population

        return self.best_solution, self.best_fitness