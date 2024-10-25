import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(10, 10 * dim // 3)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_sigma = 0.1
        self.mutation_strategy = [0.5, 1.0]
        self.learning_rate = 0.05
        self.min_learning_rate = 0.01
        self.max_learning_rate = 0.2
        self.dynamic_mutation_factor = 0.05
        self.chaos_sequence = np.random.rand(self.population_size)

    def __call__(self, func):
        evaluations = 0
        stagnation_counter = 0
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
                dynamic_mutation = self.mutation_factor + self.chaos_sequence[i] * self.dynamic_mutation_factor
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
                    stagnation_counter = 0
                else:
                    new_population[i] = self.population[i]
                    stagnation_counter += 1

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_perturbed

            if stagnation_counter > self.population_size // 2:
                self.adaptive_sigma = min(self.adaptive_sigma * 1.3, 1.0)
                self.learning_rate = min(self.learning_rate * 1.1, self.max_learning_rate)
            else:
                self.adaptive_sigma = max(self.adaptive_sigma * 0.75, 0.01)
                self.learning_rate = max(self.learning_rate * 0.85, self.min_learning_rate)

            self.chaos_sequence = (self.chaos_sequence * 3.9 * (1 - self.chaos_sequence)) % 1
            self.population = new_population

        return self.best_solution, self.best_fitness