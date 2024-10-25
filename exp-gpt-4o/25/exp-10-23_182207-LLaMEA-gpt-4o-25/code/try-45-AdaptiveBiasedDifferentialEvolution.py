import numpy as np

class AdaptiveBiasedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(10, 10 * dim // 3)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.rebalance_factor = 0.1
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.scaling_factor = 0.1
        self.mutation_strategy = [0.4, 1.2]
        self.learning_rate = 0.05
        self.min_learning_rate = 0.01
        self.max_learning_rate = 0.2
        self.dynamic_mutation_factor = 0.05

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
                biased_mutation = self.mutation_factor + np.random.normal(0, self.dynamic_mutation_factor)
                mutant = self.population[a] + biased_mutation * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])
                
                perturbation = np.random.normal(0, self.scaling_factor, self.dim)
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
                self.scaling_factor = min(self.scaling_factor * 1.3, 1.0)
                self.learning_rate = min(self.learning_rate * 1.15, self.max_learning_rate)
            else:
                self.scaling_factor = max(self.scaling_factor * 0.7, 0.01)
                self.learning_rate = max(self.learning_rate * 0.8, self.min_learning_rate)

            if evaluations < self.budget * 0.5:
                self.population += np.random.normal(0, self.rebalance_factor, self.population.shape)
                self.population = np.clip(self.population, *self.bounds)

            self.population = new_population

        return self.best_solution, self.best_fitness