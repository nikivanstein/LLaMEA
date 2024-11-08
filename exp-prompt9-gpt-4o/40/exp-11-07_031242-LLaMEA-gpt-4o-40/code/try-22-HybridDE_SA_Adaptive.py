import numpy as np

class HybridDE_SA_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 25  # Increased to enhance diversity
        self.de_mutation_factor = 0.85  # Adjusted for better exploration
        self.cr = 0.95  # Increased crossover rate
        self.initial_temperature = 100.0
        self.temperature = self.initial_temperature

    def __call__(self, func):
        np.random.seed(42)  # Changed seed for variability
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

        while evals_used < self.budget:
            for i in range(self.population_size):
                trial = de_mutation_and_crossover(i)
                trial_fitness = func(trial)
                evals_used += 1

                acceptance_probability = np.exp((fitness[i] - trial_fitness) / self.temperature)
                if trial_fitness < fitness[i] or np.random.rand() < acceptance_probability:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                # Adaptive temperature update
                if trial_fitness < fitness[i]:
                    self.temperature *= 0.95  # Faster temperature reduction
                else:
                    self.temperature += (self.initial_temperature - self.temperature) * 0.02  # Adaptive increase

                if evals_used >= self.budget:
                    break

        return best_solution, best_fitness