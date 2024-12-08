import numpy as np

class AdvancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(15, 12 * dim // 2)  # Slightly different dynamic adjustment
        self.mutation_factor = 0.9  # Adjusted mutation factor for exploration
        self.crossover_rate = 0.85  # Adjusted crossover rate to maintain diversity
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_sigma = 0.1
        self.mutation_factor_adaptive = 0.8  # Start with a different adaptive factor

    def __call__(self, func):
        evaluations = 0
        stagnation_counter = 0
        while evaluations < self.budget:
            new_population = np.empty_like(self.population)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])
                
                perturbation = np.random.normal(0, self.adaptive_sigma, self.dim)
                trial_perturbed = trial + perturbation
                trial_perturbed = np.clip(trial_perturbed, *self.bounds)
                
                trial_fitness = func(trial_perturbed)
                evaluations += 1
                
                if trial_fitness < func(self.population[i]):
                    new_population[i] = trial_perturbed
                    stagnation_counter = 0
                else:
                    new_population[i] = self.population[i]
                    stagnation_counter += 1

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_perturbed

            if stagnation_counter > self.population_size // 2:
                self.adaptive_sigma = min(self.adaptive_sigma * 1.2, 1.0)
                self.mutation_factor = min(self.mutation_factor_adaptive * 1.05, 0.95)  # Alter mutation factor adaptively
            else:
                self.adaptive_sigma = max(self.adaptive_sigma * 0.8, 0.01)
                self.mutation_factor = max(self.mutation_factor_adaptive * 0.95, 0.5)  # Adjust mutation factor back

            self.population = new_population

        return self.best_solution, self.best_fitness