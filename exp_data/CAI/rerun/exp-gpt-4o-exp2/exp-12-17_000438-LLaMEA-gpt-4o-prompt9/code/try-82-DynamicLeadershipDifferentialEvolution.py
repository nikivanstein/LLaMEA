import numpy as np

class DynamicLeadershipDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 8 * self.dim
        self.population_size = self.initial_population_size
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.CR = 0.85
        self.F = 0.75
        self.mutation_factor_bounds = (0.6, 1.1)
        self.weight_decay = 0.95
        self.leadership_prob = 0.2

    def __call__(self, func):
        evaluations = 0
        success_rate = 0.25
        while evaluations < self.budget:
            if evaluations > self.budget / 3 and self.population_size > self.dim:
                self.population_size = max(self.dim, int(self.population_size * 0.7))
                self.population = self.population[:self.population_size]

            new_population = np.zeros_like(self.population)
            elite_idx = np.argsort([func(ind) for ind in self.population])[:5]
            leaders = self.population[elite_idx]
            elite = leaders[0]

            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(leaders, 3, replace=True)
                F_dynamic = self.F * self.weight_decay * (1 + 0.2 * (success_rate - 0.3) + 0.05 * np.log1p(success_rate))
                if np.random.rand() < self.leadership_prob:  # Leadership-based mutation
                    F_dynamic = np.random.uniform(0.5, 0.9)
                F_dynamic += np.random.uniform(-0.05, 0.05)
                mutant = np.clip(a + F_dynamic * (b - c), self.lower_bound, self.upper_bound)

                CR_dynamic = self.CR * (1 - evaluations / self.budget) * (1 + 0.2 * (success_rate - 0.4))
                CR_dynamic = np.clip(CR_dynamic + 0.1 * (0.5 - success_rate), 0.1, 0.9)
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
                    success_rate = success_rate * 0.85 + 0.15
                else:
                    new_population[i] = self.population[i]
                    success_rate *= 0.85

                if evaluations >= self.budget:
                    break

            self.population = new_population
            if evaluations < self.budget:
                current_cov = np.cov(new_population.T) * (2.2**2 / self.dim + 0.01)
                proposal = np.random.multivariate_normal(self.best_solution, current_cov)
                proposal = np.clip(proposal, self.lower_bound, self.upper_bound)
                proposal_fitness = func(proposal)
                evaluations += 1
                if proposal_fitness < self.best_fitness:
                    self.best_fitness = proposal_fitness
                    self.best_solution = proposal

            if evaluations >= self.budget:
                break

            for elite in leaders:
                random_idx = np.random.randint(0, self.population_size)
                if func(elite) < func(self.population[random_idx]):
                    self.population[random_idx] = elite

        return self.best_solution, self.best_fitness