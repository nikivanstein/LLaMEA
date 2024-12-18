import numpy as np

class EnhancedHybridDifferentialEvolutionV2:
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
        self.F = 0.7
        self.mutation_factor_bounds = (0.5, 1.2)
        self.weight_decay = 0.97

    def __call__(self, func):
        evaluations = 0
        success_rate = 0.25
        dynamic_population = True
        
        while evaluations < self.budget:
            if evaluations > self.budget / 4 and self.population_size > self.dim and dynamic_population:
                self.population_size = max(self.dim, int(self.population_size * 0.7))
                self.population = self.population[:self.population_size]
                dynamic_population = False

            new_population = np.zeros_like(self.population)
            elite = self.population[np.argmin([func(ind) for ind in self.population])]
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                progress_ratio = evaluations / self.budget
                F_dynamic = self.F * ((1 - 0.4 * progress_ratio) + 0.3 * (success_rate - 0.2))
                if np.random.rand() < 0.3:
                    F_dynamic = np.random.uniform(0.5, 0.9)
                F_dynamic = np.clip(F_dynamic, self.mutation_factor_bounds[0], self.mutation_factor_bounds[1])
                mutant = np.clip(a + F_dynamic * (b - c), self.lower_bound, self.upper_bound)

                population_diversity = np.std(self.population)
                CR_dynamic = self.CR * (1 - 0.5 * progress_ratio) * (1 + 0.3 * (success_rate - 0.3) + 0.05 * population_diversity)
                CR_dynamic = np.clip(CR_dynamic, 0.1, 0.9)
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
                    success_rate = 0.7 * success_rate + 0.3
                else:
                    new_population[i] = self.population[i]
                    success_rate *= 0.7

                if evaluations >= self.budget:
                    break

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

            self.population = new_population
            elite_index = np.random.randint(0, self.population_size)
            self.population[elite_index] = elite
            for _ in range(2):
                random_index = np.random.randint(0, self.population_size)
                if func(elite) < func(self.population[random_index]):
                    self.population[random_index] = elite

        return self.best_solution, self.best_fitness