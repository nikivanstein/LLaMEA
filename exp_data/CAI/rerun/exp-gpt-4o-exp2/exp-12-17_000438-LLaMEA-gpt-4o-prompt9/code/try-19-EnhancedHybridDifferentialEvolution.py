import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 12 * self.dim
        self.population_size = self.initial_population_size
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.CR = 0.85  # Crossover rate
        self.F = 0.75   # Differential weight
        self.mutation_factor_bounds = (0.4, 0.9)
        self.weight_decay = 0.95  # Exponential weight decay factor

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            # Dynamic population sizing based on remaining budget
            if evaluations > self.budget / 3 and self.population_size > self.dim:
                self.population_size = max(self.dim, self.population_size // 2)
                self.population = self.population[:self.population_size]

            new_population = np.zeros_like(self.population)
            elite = np.copy(self.population[np.argmin([func(ind) for ind in self.population])])  # Retain best
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                selected = np.random.choice(indices, 3, replace=False)
                a, b, c = self.population[selected]
                F_dynamic = self.F * self.weight_decay + (np.random.rand() - 0.5) * 0.1
                if np.random.rand() < 0.2:  # Adaptive mutation threshold
                    F_dynamic = np.random.uniform(*self.mutation_factor_bounds)
                mutant = np.clip(a + F_dynamic * (b - c), self.lower_bound, self.upper_bound)

                CR_dynamic = self.CR + (0.1 * (self.budget - evaluations) / self.budget)
                cross_points = np.random.rand(self.dim) < CR_dynamic
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial
                if trial_fitness < func(self.population[i]) or self._crowding_distance(self.population[i], trial) < 0.1:
                    new_population[i] = trial
                else:
                    new_population[i] = self.population[i]

                if evaluations >= self.budget:
                    break

            if evaluations < self.budget:
                current_cov = np.cov(new_population.T) * (2.1**2 / self.dim + 0.01)
                proposal = np.random.multivariate_normal(self.best_solution, current_cov)
                proposal = np.clip(proposal, self.lower_bound, self.upper_bound)
                proposal_fitness = func(proposal)
                evaluations += 1
                if proposal_fitness < self.best_fitness:
                    self.best_fitness = proposal_fitness
                    self.best_solution = proposal

            if evaluations >= self.budget:
                break

            self.population = new_population
            self.population[np.random.randint(self.population_size)] = elite

            # Local Search Intensification
            if np.random.rand() < 0.1:
                local_search_point = elite + np.random.randn(self.dim) * 0.05
                local_search_point = np.clip(local_search_point, self.lower_bound, self.upper_bound)
                local_fitness = func(local_search_point)
                evaluations += 1
                if local_fitness < self.best_fitness:
                    self.best_fitness = local_fitness
                    self.best_solution = local_search_point

        return self.best_solution, self.best_fitness

    def _crowding_distance(self, individual, trial):
        return np.linalg.norm(trial - individual) / np.linalg.norm(self.upper_bound - self.lower_bound)