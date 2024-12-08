import numpy as np

class HybridDE_AdaptiveSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.population_size = 10  # Initial population size
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.initial_temp = 1.0  # Initial temperature for simulated annealing
        self.cooling_rate = 0.95  # Adjusted cooling rate for simulated annealing

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Generate a mutant vector with adaptive scaling
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                F_adaptive = self.F * np.random.rand()  # Adaptive factor
                mutant = np.clip(a + F_adaptive * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                self.evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Adaptive Simulated Annealing acceptance
                current_temp = self.initial_temp * (self.cooling_rate ** (self.evaluations / self.budget))
                if trial_fitness >= fitness[i]:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / current_temp)
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness

            # Dynamic parameter adjustment
            self.F = 0.7 + 0.3 * np.random.rand()
            self.CR = 0.85 + 0.15 * np.random.rand()

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]