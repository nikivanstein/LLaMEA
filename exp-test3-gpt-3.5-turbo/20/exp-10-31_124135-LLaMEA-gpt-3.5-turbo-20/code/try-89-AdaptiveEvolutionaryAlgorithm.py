import numpy as np

class AdaptiveEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 0.1  # Initial mutation rate

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evals = 0
        while evals < self.budget:
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            
            # Adaptive mutation operator based on Gaussian distribution with dynamic mutation rate
            mutated_population = population + np.random.normal(0, self.mutation_rate, (self.budget, self.dim))
            population = mutated_population
            evals += self.budget
            
            # Update mutation rate based on the best fitness value
            self.mutation_rate = max(0.01, 0.1 / np.sqrt(np.mean(fitness) + 1e-9))

        return best_solution