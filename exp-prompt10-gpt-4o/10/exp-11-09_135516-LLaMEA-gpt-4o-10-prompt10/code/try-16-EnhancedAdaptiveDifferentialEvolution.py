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

        initial_pop = np.copy(self.population)
        twin_population = np.copy(self.population)  # Twin population for diversity

        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.fitness[i] < best_fitness:
                best_fitness = self.fitness[i]
                best_solution = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                F = 0.4 + np.random.rand() * 0.6  # Slightly broadened range for F
                CR = 0.7 + np.random.rand() * 0.3  # Slightly broadened range for CR

                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < CR
                trial[crossover_points] = mutant[crossover_points]

                if np.random.rand() < 0.1:  # Occasionally use twin population
                    twin_index = np.random.randint(self.population_size)
                    trial = np.clip(trial + 0.1 * (twin_population[twin_index] - trial), self.lower_bound, self.upper_bound)

                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial
                if self.evaluations >= self.budget:
                    break

            # Update twin population occasionally to maintain diversity
            if self.evaluations % (self.budget // 10) == 0:
                twin_population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

        return best_solution