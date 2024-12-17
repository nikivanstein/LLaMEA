import numpy as np

class ImprovedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * self.dim
        self.population_size = self.initial_population_size
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.CR = 0.9
        self.F = 0.8
        self.mutation_factor_bounds = (0.5, 1.2)
        self.weight_decay = 0.97
        self.success_rate = 0.3

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            if evaluations > self.budget / 4 and self.population_size > self.dim:
                self.population_size = max(self.dim, int(self.population_size * 0.7))
                self.population = self.population[:self.population_size]

            new_population = np.zeros_like(self.population)
            elite_index = np.argmin([func(ind) for ind in self.population])
            elite = np.copy(self.population[elite_index])

            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                dynamic_F = self.F * (1 + 0.3 * (self.success_rate - 0.5))
                mutant = np.clip(a + dynamic_F * (b - c), self.lower_bound, self.upper_bound)

                if np.random.rand() < 0.5:
                    dynamic_CR = self.CR * (1 - evaluations / self.budget)
                else:
                    dynamic_CR = np.random.uniform(0.5, 0.9)

                cross_points = np.random.rand(self.dim) < dynamic_CR
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
                    self.success_rate = self.success_rate * 0.85 + 0.15
                else:
                    new_population[i] = self.population[i]
                    self.success_rate *= 0.85

                if evaluations >= self.budget:
                    break

            if evaluations < self.budget:
                current_cov = np.cov(new_population.T) * (2.0**2 / self.dim + 0.01)
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
            self.population[np.random.randint(0, self.population_size)] = elite

        return self.best_solution, self.best_fitness