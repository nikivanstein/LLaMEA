import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4 * self.dim, 20)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.diversity_threshold = 0.1 * (self.upper_bound - self.lower_bound)

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
                F = 0.5 + np.random.rand() * 0.4  # Adjusted range for mutation factor
                CR = 0.6 + np.random.rand() * 0.4  # Adjusted range for crossover rate

                # Mutation
                indices = np.random.choice([x for x in range(self.population_size) if x != i], 5, replace=False)
                base = self.population[indices[0]]
                diff1 = self.population[indices[1]] - self.population[indices[2]]
                diff2 = self.population[indices[3]] - self.population[indices[4]]
                mutant = np.clip(base + F * (diff1 + diff2), self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < CR
                if not np.any(crossover_points):
                    crossover_points[np.random.randint(self.dim)] = True
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

                # Diversity preservation mechanism
                if np.std(self.population) < self.diversity_threshold:
                    self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                    self.fitness = np.full(self.population_size, np.inf)
                    for j in range(self.population_size):
                        self.fitness[j] = func(self.population[j])
                        self.evaluations += 1
                        if self.fitness[j] < best_fitness:
                            best_fitness = self.fitness[j]
                            best_solution = self.population[j]

                if self.evaluations >= self.budget:
                    break
        return best_solution