import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.mutation_f = 0.6  # Adjusted for improved initial search diversity
        self.crossover_prob = 0.8  # Slightly increased to enhance diversity
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim)
        )
        self.fitness = np.full(self.population_size, np.inf)
        self.used_budget = 0
        self.best_fitness = np.inf

    def __call__(self, func):
        self.evaluate_population(func)
        stagnation_counter = 0
        
        while self.used_budget < self.budget:
            for i in range(self.population_size):
                if self.used_budget >= self.budget:
                    break

                best_idx = np.argmin(self.fitness)
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mu = self.mutation_f * (b - c) + 0.5 * (self.population[best_idx] - a)
                mutant = np.clip(a + mu, self.lower_bound, self.upper_bound)
                
                trial = np.copy(self.population[i])
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial[cross_points] = mutant[cross_points]

                trial_fitness = func(trial)
                self.used_budget += 1
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial
                    stagnation_counter = 0  # Reset counter on improvement
                else:
                    stagnation_counter += 1

            if stagnation_counter > self.population_size * 2:
                self.crossover_prob = 0.9
                self.mutation_f *= 1.1
                stagnation_counter = 0

            if self.used_budget > self.budget * 0.6 and self.population_size > self.dim:
                self.population_size = int(self.initial_population_size * 0.7)
                self.population = self.population[:self.population_size]
                self.fitness = self.fitness[:self.population_size]

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.used_budget >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.used_budget += 1