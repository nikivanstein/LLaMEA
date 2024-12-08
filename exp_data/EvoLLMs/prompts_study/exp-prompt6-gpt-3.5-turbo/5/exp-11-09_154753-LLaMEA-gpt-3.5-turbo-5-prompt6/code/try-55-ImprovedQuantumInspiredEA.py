import numpy as np

class ImprovedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_rate = 1.0
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            parents = self.population[np.argsort(fitness)[:2]]
            offspring = 0.5 * (parents[0] + parents[1]) + np.random.normal(0, self.mutation_rate, self.dim)  # Adjusted mutation rate
            worst_idx = np.argmax(fitness)
            self.population[worst_idx] = offspring
            best_fitness = min(fitness)
            convergence_rate = np.mean([(best_fitness - f) for f in fitness])  # Calculate convergence rate
            self.mutation_rate = max(0.1, self.mutation_rate * (1 - 0.5 * convergence_rate))  # Dynamic mutation rate adjustment
        return self.population[np.argmin([func(individual) for individual in self.population])]