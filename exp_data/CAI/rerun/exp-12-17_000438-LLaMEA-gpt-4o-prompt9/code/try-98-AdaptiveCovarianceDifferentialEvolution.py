import numpy as np

class AdaptiveCovarianceDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 6 * self.dim
        self.population_size = self.initial_population_size
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.CR = 0.85
        self.F = 0.65
        self.mutation_factor_bounds = (0.55, 1.0)
        self.weight_decay = 0.95

    def __call__(self, func):
        evaluations = 0
        success_rate = 0.3
        while evaluations < self.budget:
            if evaluations > self.budget / 2 and self.population_size > self.dim:
                self.population_size = max(self.dim, int(self.population_size * 0.7))
                self.population = self.population[:self.population_size]

            new_population = np.zeros_like(self.population)
            elite = np.copy(self.population[np.argmin([func(ind) for ind in self.population])])

            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                progress_ratio = evaluations / self.budget
                F_dynamic = self.F * (1 - 0.4 * progress_ratio) * self.weight_decay * (1 + 0.2 * (success_rate - 0.25) + 0.05 * np.log1p(success_rate))
                if np.random.rand() < 0.3:
                    F_dynamic = np.random.uniform(0.4, 0.8)
                F_dynamic += np.random.uniform(-0.04, 0.04)
                mutant = np.clip(a + F_dynamic * (b - c), self.lower_bound, self.upper_bound)

                population_diversity = np.std(self.population)
                CR_dynamic = self.CR * (1 - evaluations / self.budget) * (1 + 0.2 * (success_rate - 0.35) + 0.05 * population_diversity)
                CR_dynamic = np.clip(CR_dynamic + 0.12 * (0.5 - success_rate), 0.1, 0.95)
                cross_points = np.random.rand(self.dim) < CR_dynamic
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
                    success_rate = success_rate * 0.9 + 0.1
                else:
                    new_population[i] = self.population[i]
                    success_rate *= 0.9

                if evaluations >= self.budget:
                    break

            if evaluations < self.budget:
                current_cov = np.cov(new_population.T) * (2.6**2 / self.dim + 0.02)
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
            elite_index = np.random.randint(0, self.population_size)
            self.population[elite_index] = elite
            for _ in range(2):
                random_index = np.random.randint(0, self.population_size)
                if func(elite) < func(self.population[random_index]):
                    self.population[random_index] = elite

        return self.best_solution, self.best_fitness