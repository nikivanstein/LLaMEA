import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(10, 10 * dim // 3)
        self.mutation_factor = 0.9  # Altered mutation factor for diversity
        self.crossover_rate = 0.8  # Adjusted crossover rate for exploration
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_sigma = 0.15  # Adjusted for increased local search when needed
        self.mutation_strategy = [0.6, 1.2]  # Diverse choice of mutation strategies
        self.learning_rate = 0.06  # Enhanced learning rate
        self.min_learning_rate = 0.02
        self.max_learning_rate = 0.25
        self.dynamic_mutation_factor = 0.07  # Increased for better exploration

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
                dynamic_mutation = self.mutation_factor + np.random.normal(0, self.dynamic_mutation_factor)
                mutant = self.population[a] + np.random.choice(self.mutation_strategy) * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])
                
                perturbation = np.random.normal(0, self.adaptive_sigma, self.dim) * np.random.choice([-1, 1], size=self.dim)  # Added stochastic sign for perturbation
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
                self.adaptive_sigma = min(self.adaptive_sigma * 1.25, 1.0)  # Enhanced adaptive increase
                self.learning_rate = min(self.learning_rate * 1.15, self.max_learning_rate)  # Enhanced learning rate adaptability
            else:
                self.adaptive_sigma = max(self.adaptive_sigma * 0.75, 0.01)  # Enhanced adaptive decrease
                self.learning_rate = max(self.learning_rate * 0.85, self.min_learning_rate)  # Enhanced learning rate adaptability

            self.population = new_population

        return self.best_solution, self.best_fitness