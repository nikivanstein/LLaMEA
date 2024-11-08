import numpy as np

class EnhancedDE_ASA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 25  # Increased population size
        self.de_mutation_factor = 0.9  # Adjusted mutation factor
        self.cr = 0.8  # Adjusted crossover rate
        self.initial_temperature = 150.0  # Higher initial temperature

    def __call__(self, func):
        np.random.seed(0)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals_used = self.population_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        def de_mutation_and_crossover(target_idx):
            indices = list(range(self.population_size))
            indices.remove(target_idx)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.de_mutation_factor * (b - c), self.lower_bound, self.upper_bound)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[target_idx])
            return trial

        temperature = self.initial_temperature
        while evals_used < self.budget:
            for i in range(self.population_size):
                trial = de_mutation_and_crossover(i)
                trial_fitness = func(trial)
                evals_used += 1

                # Adaptive Simulated Annealing acceptance
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                # Update temperature and parameters dynamically
                temperature *= 0.98  # Faster cooling schedule
                self.de_mutation_factor = 0.5 + 0.4 * (self.budget - evals_used) / self.budget
                self.cr = 0.6 + 0.3 * (evals_used / self.budget)

                if evals_used >= self.budget:
                    break

        return best_solution, best_fitness