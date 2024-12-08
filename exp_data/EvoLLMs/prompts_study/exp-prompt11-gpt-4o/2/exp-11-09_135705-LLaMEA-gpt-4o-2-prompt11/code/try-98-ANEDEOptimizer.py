import numpy as np

class ANEDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(4, int(0.5 * self.budget / self.dim))
        self.F = 0.85
        self.CR = 0.85
        self.refresh_rate = 0.1  # Added population refresh rate
        self.best_solution = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size
        iteration = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                dynamic_F = self.F * (1 + np.random.uniform(-0.1, 0.2))
                mutant = np.clip(a + dynamic_F * (b - c), self.lower_bound, self.upper_bound)

                adaptive_CR = self.CR + 0.05 * np.random.uniform(-1, 1 + iteration / 1000.0)  # Dynamic CR adjustment
                trial = np.copy(population[i])
                crossover_mask = np.random.rand(self.dim) < adaptive_CR
                trial[crossover_mask] = mutant[crossover_mask]

                if np.random.rand() < 0.2:
                    local_center = np.median(population, axis=0)
                    trial = np.clip(trial + np.random.uniform(-0.1, 0.1, self.dim) * (local_center - trial), self.lower_bound, self.upper_bound)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

            if (iteration + 1) % int(1 / self.refresh_rate) == 0:  # Periodic refresh of a portion of the population
                population[:self.population_size // 5] = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size // 5, self.dim))
                fitness[:self.population_size // 5] = np.apply_along_axis(func, 1, population[:self.population_size // 5])
                evaluations += self.population_size // 5

            iteration += 1

        return self.best_solution, self.best_fitness