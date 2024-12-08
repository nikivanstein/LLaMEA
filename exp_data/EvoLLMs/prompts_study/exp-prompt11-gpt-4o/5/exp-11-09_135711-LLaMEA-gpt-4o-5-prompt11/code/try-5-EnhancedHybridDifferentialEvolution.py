import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(20, min(100, 20 + dim * 5))  # Adjust initial pop size
        self.mutation_factor = 0.5
        self.crossover_probability = 0.9

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = population[np.random.choice(candidates, 3, replace=False)]

                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Adaptive parameter control and strategy adjustment
            if eval_count % (self.population_size * 5) == 0:
                self.mutation_factor = 0.4 + np.random.rand() * 0.2
                self.crossover_probability = 0.8 + np.random.rand() * 0.2
                self.population_size = max(20, int(self.population_size * (0.9 + np.random.rand() * 0.2)))
                population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                fitness = np.array([func(ind) for ind in population[:self.population_size]])
                eval_count = self.population_size

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]