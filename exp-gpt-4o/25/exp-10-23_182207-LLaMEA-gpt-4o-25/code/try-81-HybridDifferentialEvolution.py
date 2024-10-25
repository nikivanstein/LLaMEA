import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(10, 15 * dim // 3)  # Adjusted population size for enhanced diversity
        self.mutation_factor = 0.7  # Slightly reduced to balance exploration
        self.crossover_rate = 0.85  # Slightly reduced to increase local search potential
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_sigma = 0.15  # Increased for stronger perturbation effects
        self.mutation_strategy = [0.4, 0.9]  # Adjusted for diversity in mutation strength
        self.learning_rate = 0.06  # Slightly increased for faster convergence
        self.min_learning_rate = 0.015  # Adjusted for stability
        self.max_learning_rate = 0.25  # Increased for aggressive search
        self.dynamic_mutation_factor = 0.07  # Increased for greater effect in dynamic mutation
        self.dynamic_population = True  # New flag for dynamic population resizing

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
                if self.dynamic_population:
                    self.population_size = min(self.population_size + 5, self.budget // self.dim)
                    new_individuals = np.random.uniform(self.bounds[0], self.bounds[1], (5, self.dim))
                    new_population = np.vstack((new_population, new_individuals))
            else:
                self.adaptive_sigma = max(self.adaptive_sigma * 0.75, 0.01)
                self.learning_rate = max(self.learning_rate * 0.85, self.min_learning_rate)
                if self.dynamic_population and self.population_size > 10:
                    self.population_size -= 5
                    new_population = new_population[:self.population_size]

            self.population = new_population[:self.population_size]
        
        return self.best_solution, self.best_fitness