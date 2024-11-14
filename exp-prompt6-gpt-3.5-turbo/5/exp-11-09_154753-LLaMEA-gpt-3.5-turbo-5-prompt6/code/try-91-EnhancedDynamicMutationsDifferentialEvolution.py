import numpy as np

class EnhancedDynamicMutationsDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_factors = np.full(budget, 0.8)
        self.best_fitness = np.inf

    def __call__(self, func):
        crossover_rate = 0.9
        for _ in range(self.budget):
            diversity_factor = np.std(self.population, axis=0)  # Measure population diversity
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                scaling_factor = 0.8 + self.mutation_factors[i] * np.mean(diversity_factor)  # Adjust mutation step size dynamically
                mutant_vector = self.population[a] + scaling_factor * (self.population[b] - self.population[c])
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                trial_fitness = func(trial_vector)
                if trial_fitness < func(self.population[i]):
                    self.population[i] = trial_vector
                    self.mutation_factors[i] = min(self.mutation_factors[i] * (1 + 0.05 * (self.best_fitness - trial_fitness) / self.best_fitness), 5.0)  # Adaptive mutation step size based on individual performance
                    self.best_fitness = min(self.best_fitness, trial_fitness)  # Update best fitness
        return self.population[np.argmin([func(individual) for individual in self.population])]