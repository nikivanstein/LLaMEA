import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 8 * self.dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0

    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf

        # Initial fitness evaluation
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.fitness[i] < best_fitness:
                best_fitness = self.fitness[i]
                best_solution = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Dynamic adaptation of mutation factor and crossover rate with local search consideration
                F = 0.4 + np.random.rand() * 0.6  # Randomly vary between 0.4 and 1.0
                CR = 0.7 + np.random.rand() * 0.3  # Randomly vary between 0.7 and 1.0

                # Mutation
                indices = np.random.choice(self.population_size, 5, replace=False)
                a, b, c, d, e = self.population[indices]
                mutant = np.clip(a + F * (b - c) + F * (d - e), self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < CR
                trial[crossover_points] = mutant[crossover_points]

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial
                    
                # Local search in nearby explored areas
                if np.random.rand() < 0.1:
                    local_trial = np.copy(best_solution + np.random.normal(0, 0.1, self.dim))
                    local_trial = np.clip(local_trial, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_trial)
                    self.evaluations += 1
                    if local_fitness < best_fitness:
                        best_fitness = local_fitness
                        best_solution = local_trial

                if self.evaluations >= self.budget:
                    break
        return best_solution