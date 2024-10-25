import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(15, 12 * dim // 2)
        self.mutation_factor = 0.9
        self.crossover_rate = 0.85
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_sigma = 0.15
        self.niche_radius = 0.5
        self.success_rate = 0.1
        self.dynamic_sigma_factor = 0.03

    def __call__(self, func):
        evaluations = 0
        success_count = 0
        while evaluations < self.budget:
            new_population = np.empty_like(self.population)
            fitness_values = np.apply_along_axis(func, 1, self.population)
            evaluations += self.population_size
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                dynamic_mutation = self.mutation_factor + np.random.normal(0, self.dynamic_sigma_factor)
                mutant = self.population[a] + dynamic_mutation * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])
                
                perturbation = np.random.normal(0, self.adaptive_sigma, self.dim)
                trial_perturbed = trial + np.random.normal(0, self.niche_radius) * perturbation
                trial_perturbed = np.clip(trial_perturbed, *self.bounds)
                
                trial_fitness = func(trial_perturbed)
                evaluations += 1

                if trial_fitness < fitness_values[i]:
                    new_population[i] = trial_perturbed
                    success_count += 1
                else:
                    new_population[i] = self.population[i]

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_perturbed

            if success_count / self.population_size > self.success_rate:
                self.adaptive_sigma = min(self.adaptive_sigma * 1.2, 1.0)
                self.niche_radius = min(self.niche_radius * 1.05, 1.0)
            else:
                self.adaptive_sigma = max(self.adaptive_sigma * 0.8, 0.01)
                self.niche_radius = max(self.niche_radius * 0.95, 0.1)

            self.population = new_population

        return self.best_solution, self.best_fitness