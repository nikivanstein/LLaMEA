import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(10, 10 * dim // 2)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_step = 0.1

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
                
                # Dual mutation strategy
                if np.random.rand() > 0.5:
                    mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
                else:
                    mutant = self.population[a] + self.mutation_factor * (self.population[a] - self.population[c])

                mutant = np.clip(mutant, *self.bounds)

                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])

                # Adaptive step size
                step_size = np.random.normal(0, self.adaptive_step, self.dim)
                trial_perturbed = trial + step_size
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
                self.adaptive_step = min(self.adaptive_step * 1.1, 1.0)
            else:
                self.adaptive_step = max(self.adaptive_step * 0.9, 0.01)

            self.population = new_population

        return self.best_solution, self.best_fitness