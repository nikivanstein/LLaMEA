import numpy as np

class EnhancedHybridDE_SA_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 40  # Increased initial population
        self.min_population_size = 8  # Reduced minimum population size
        self.initial_de_mutation_factor = 0.5  # Dynamic mutation factor start
        self.cr_min = 0.65  # Crossover rate range start
        self.cr_max = 0.95  # Crossover rate range end
        self.initial_temperature = 100.0  # Higher initial temperature
        self.temperature_decay = 0.90  # Slower decay

    def __call__(self, func):
        np.random.seed(0)
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals_used = population_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        temperature = self.initial_temperature
        while evals_used < self.budget:
            de_mutation_factor = self.initial_de_mutation_factor + (1 - self.initial_de_mutation_factor) * (evals_used / self.budget)
            for i in range(population_size):
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                cr = self.cr_min + (self.cr_max - self.cr_min) * (evals_used / self.budget)
                mutant = np.clip(a + de_mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evals_used += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evals_used >= self.budget:
                    break

            temperature *= self.temperature_decay

            population_size = max(self.min_population_size, int(self.initial_population_size * (1 - evals_used / self.budget)))
            population = population[:population_size]
            fitness = fitness[:population_size]

        return best_solution, best_fitness