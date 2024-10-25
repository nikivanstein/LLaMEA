import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(12, 8 * dim // 3)
        self.mutation_factor = 0.7
        self.crossover_rate = 0.85
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_sigma = 0.2
        self.mutation_strategy = [0.5, 1.2]
        self.learning_rate = 0.07
        self.min_learning_rate = 0.02
        self.max_learning_rate = 0.15
        self.dynamic_mutation_factor = 0.03

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            new_population = np.empty_like(self.population)
            fitness_values = np.apply_along_axis(func, 1, self.population)
            evaluations += self.population_size
            diversity = np.std(self.population, axis=0)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                dynamic_mutation = self.mutation_factor + np.random.normal(0, self.dynamic_mutation_factor)
                mutant = self.population[a] + np.random.choice(self.mutation_strategy) * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])
                
                perturbation = np.random.normal(0, self.adaptive_sigma, self.dim)
                trial_perturbed = trial + self.learning_rate * perturbation
                trial_perturbed = np.clip(trial_perturbed, *self.bounds)
                
                trial_fitness = func(trial_perturbed)
                evaluations += 1

                if trial_fitness < fitness_values[i]:
                    new_population[i] = trial_perturbed
                else:
                    new_population[i] = self.population[i]

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_perturbed

            self.adaptive_sigma = min(max(0.1, 0.7 * np.mean(diversity)), 0.3)
            self.learning_rate = min(max(self.learning_rate * 0.9, self.min_learning_rate), self.max_learning_rate)

            self.population = new_population

        return self.best_solution, self.best_fitness