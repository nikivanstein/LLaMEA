import numpy as np

class FireflyDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.num_fireflies = 10
        self.F = 0.5
        self.CR = 0.9

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

        def evaluate_population(population):
            return np.array([func(solution) for solution in population])

        def update_population(population, fitness):
            for i in range(self.num_fireflies):
                firefly = population[i]
                for j in range(self.pop_size):
                    r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                    mutant = population[r1] + self.F * (population[r2] - population[r3])
                    crossover_mask = np.random.rand(self.dim) < self.CR
                    trial = np.where(crossover_mask, mutant, firefly)

                    if func(trial) < fitness[i]:
                        population[i] = trial
                        fitness[i] = func(trial)

            return population, fitness

        population = initialize_population()
        fitness = evaluate_population(population)

        for _ in range(self.budget - self.budget // 10):
            population, fitness = update_population(population, fitness)

        best_idx = np.argmin(fitness)
        return population[best_idx]