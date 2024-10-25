import numpy as np

class AdvancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(15, 10 * dim // 2)  # Slightly larger dynamic adjustment
        self.mutation_factor = 0.85  # Slightly increased mutation factor
        self.crossover_rate = 0.85  # Slightly decreased crossover rate
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_sigma = 0.05  # Reduced initial sigma for Gaussian perturbation
        self.fitness_values = np.full(self.population_size, np.inf)  # Track fitness values for diversity

    def __call__(self, func):
        evaluations = 0
        stagnation_counter = 0
        while evaluations < self.budget:
            new_population = np.empty_like(self.population)
            new_fitness_values = np.empty(self.population_size)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])
                
                # Adaptive Gaussian perturbation
                perturbation = np.random.normal(0, self.adaptive_sigma, self.dim)
                trial_perturbed = trial + perturbation
                trial_perturbed = np.clip(trial_perturbed, *self.bounds)
                
                trial_fitness = func(trial_perturbed)
                evaluations += 1

                if trial_fitness < self.fitness_values[i]:
                    new_population[i] = trial_perturbed
                    new_fitness_values[i] = trial_fitness
                    stagnation_counter = 0
                else:
                    new_population[i] = self.population[i]
                    new_fitness_values[i] = self.fitness_values[i]
                    stagnation_counter += 1

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_perturbed

            self.population = new_population
            self.fitness_values = new_fitness_values

            # Adjust perturbation based on stagnation
            if stagnation_counter > self.population_size // 3:
                self.adaptive_sigma = min(self.adaptive_sigma * 1.15, 1.0)  # Increase sigma
            else:
                self.adaptive_sigma = max(self.adaptive_sigma * 0.9, 0.01)  # Decrease sigma

        return self.best_solution, self.best_fitness