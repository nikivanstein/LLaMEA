import numpy as np

class EnhancedDifferentialEvolutionAdaptiveCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * self.dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.CR = 0.9
        self.F = 0.8
        self.mutation_factor_bounds = (0.5, 1.0)
        self.evaluation_ratio = 0.7  # Adaptive population resizing parameter

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            new_population = np.zeros_like(self.population)
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                F_dynamic = self.F + (np.random.rand() - 0.5) * 0.1
                if np.random.rand() < 0.2:
                    F_dynamic = np.random.uniform(*self.mutation_factor_bounds)
                mutant = np.clip(a + F_dynamic * (b - c), self.lower_bound, self.upper_bound)
                
                # Novel Crossover Strategy: adjusting CR dynamically
                cross_points = np.random.rand(self.dim) < (self.CR + 0.1 * np.sin(evaluations / self.budget * np.pi))
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

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

            # Adaptive Population Resizing
            if evaluations < self.budget:
                current_cov = np.cov(new_population.T) * 2.4**2 / self.dim
                proposal = np.random.multivariate_normal(self.best_solution, current_cov)
                proposal = np.clip(proposal, self.lower_bound, self.upper_bound)
                proposal_fitness = func(proposal)
                evaluations += 1
                if proposal_fitness < self.best_fitness:
                    self.best_fitness = proposal_fitness
                    self.best_solution = proposal
                # Resize the population according to evaluation ratio
                self.population_size = max(4, int(self.population_size * self.evaluation_ratio))

            if evaluations >= self.budget:
                break

            self.population = new_population

        return self.best_solution, self.best_fitness