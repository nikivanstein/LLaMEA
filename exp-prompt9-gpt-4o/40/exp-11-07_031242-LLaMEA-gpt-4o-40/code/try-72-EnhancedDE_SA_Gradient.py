import numpy as np

class EnhancedDE_SA_Gradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 30  # Slight increase for diversity
        self.min_population_size = 10  # Maintain minimum diversity
        self.de_mutation_factor = 0.75  # Slightly increased for better exploration
        self.cr = 0.8  # Adjusted crossover rate
        self.initial_temperature = 120.0  # Higher initial temperature
        self.temperature_decay = 0.9
        self.gradient_step = 0.01  # Gradient-inspired step size

    def __call__(self, func):
        np.random.seed(42)
        population_size = self.initial_population_size
        # Chaotic initialization using log-uniform distribution
        population = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.lognormal(0, 1, (population_size, self.dim))
        population = np.clip(population, self.lower_bound, self.upper_bound)
        fitness = np.array([func(ind) for ind in population])
        evals_used = population_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        def de_mutation_and_crossover(target_idx):
            indices = list(range(population_size))
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
            for i in range(population_size):
                trial = de_mutation_and_crossover(i)
                # Gradient-inspired local search
                local_trial = np.clip(trial + self.gradient_step * np.sign(best_solution - trial), self.lower_bound, self.upper_bound)
                trial_fitness = func(local_trial)
                evals_used += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i] = local_trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = local_trial
                        best_fitness = trial_fitness

                temperature *= self.temperature_decay

                if evals_used >= self.budget:
                    break

            # Adaptive population size control
            population_size = max(self.min_population_size, int(self.initial_population_size * (1 - evals_used / self.budget)))
            population = population[:population_size]
            fitness = fitness[:population_size]

        return best_solution, best_fitness