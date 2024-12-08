import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def evolve(self, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        elite = self.population[sorted_indices[0]]

        # Incorporating Differential Evolution mutation strategy
        F = 0.5  # Weight factor for mutation
        for i in range(1, self.budget):
            mutant = self.population[np.random.choice(self.budget, 3, replace=False)]
            trial_vector = self.population[i] + F * (mutant[0] - mutant[1])
            crossover_points = np.random.rand(self.dim) < 0.5
            self.population[i] = np.where(crossover_points, trial_vector, self.population[i])

    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]

        for _ in range(self.budget):
            self.evolve(fitness_values)
            fitness_values = [func(ind) for ind in self.population]

        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution