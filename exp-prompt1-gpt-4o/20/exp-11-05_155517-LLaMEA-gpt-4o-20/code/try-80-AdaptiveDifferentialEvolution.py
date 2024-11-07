import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.5 + np.random.rand(self.population_size) * 0.5  # Dynamic mutation factor
        self.crossover_probability = 0.9

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        evaluations = self.population_size
        success_rate = np.zeros(self.population_size)

        def crowding_distance(individual):
            return np.sum(np.square(individual))

        while evaluations < self.budget:
            adaptive_pop_size = int(self.population_size * (1.0 - 0.5 * (evaluations / self.budget)))
            ranked_indices = np.argsort(fitness)
            elite_size = max(1, int(0.1 * self.population_size))
            elite_indices = ranked_indices[:elite_size]

            new_population = []
            for i in range(adaptive_pop_size):
                if evaluations >= self.budget:
                    break
                
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = self.mutation_factor[np.random.randint(self.population_size)]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                adaptive_crossover = min(1.0, self.crossover_probability + 0.1 * (1 - success_rate[i]))
                cross_points = np.random.rand(self.dim) < adaptive_crossover

                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    new_population.append((trial, f_trial))
                    success_rate[i] = 1
                else:
                    success_rate[i] = 0.9 * success_rate[i]

            if new_population:
                j = 0
                for idx, fit in sorted(new_population, key=lambda x: x[1]):
                    population[ranked_indices[j]] = idx
                    fitness[ranked_indices[j]] = fit
                    j += 1

        best_idx = np.argmin(fitness)
        return population[best_idx]