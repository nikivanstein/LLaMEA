import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * self.dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.CR = 0.8  # Crossover rate
        self.F = 0.9   # Differential weight
        self.mutation_factor_bounds = (0.4, 0.9)
        self.local_search_radius = 0.1

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            new_population = np.zeros_like(self.population)
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                F_dynamic = self.F + (np.random.rand() - 0.5) * 0.1
                if np.random.rand() < 0.25:  # Adjusted mutation threshold
                    F_dynamic = np.random.uniform(*self.mutation_factor_bounds)
                mutant = np.clip(a + F_dynamic * (b - c), self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < self.CR
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

            if evaluations < self.budget:
                current_cov = np.cov(new_population.T) * 2.4**2 / self.dim
                proposal = np.random.multivariate_normal(self.best_solution, current_cov)
                proposal = np.clip(proposal, self.lower_bound, self.upper_bound)
                proposal_fitness = func(proposal)
                evaluations += 1
                if proposal_fitness < self.best_fitness:
                    self.best_fitness = proposal_fitness
                    self.best_solution = proposal

            if evaluations < self.budget:
                for j in range(self.population_size):
                    local_search_solution = new_population[j] + np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
                    local_search_solution = np.clip(local_search_solution, self.lower_bound, self.upper_bound)
                    local_search_fitness = func(local_search_solution)
                    evaluations += 1
                    if local_search_fitness < func(new_population[j]):
                        new_population[j] = local_search_solution
                    if evaluations >= self.budget:
                        break

            if evaluations >= self.budget:
                break

            self.population = new_population

        return self.best_solution, self.best_fitness