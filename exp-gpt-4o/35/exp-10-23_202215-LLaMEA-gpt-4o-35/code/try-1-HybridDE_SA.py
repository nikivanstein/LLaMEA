import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.initial_population_size = 10
        self.F = 0.8
        self.CR = 0.9
        self.initial_temp = 1.0
        self.cooling_rate = 0.995
        self.evaluations = 0

    def __call__(self, func):
        # Initialize population with an adaptive size
        population_size = self.initial_population_size + int(self.budget / 100)
        population = np.random.uniform(self.bounds[0], self.bounds[1], (population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations = population_size

        while self.evaluations < self.budget:
            for i in range(population_size):
                # Generate a mutant vector
                a, b, c = population[np.random.choice(population_size, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover with a dynamic strategy
                crossover_mask = np.random.rand(self.dim) < (self.CR / 2 + self.CR / 2 * np.random.rand())
                trial = np.where(crossover_mask, mutant, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                self.evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Simulated Annealing acceptance
                current_temp = self.initial_temp * (self.cooling_rate ** (self.evaluations / self.budget))
                if trial_fitness >= fitness[i]:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / current_temp)
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness

            # Adaptive parameter adjustment
            self.F = 0.7 + 0.3 * np.random.rand()
            self.CR = 0.8 + 0.2 * np.random.rand()

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]