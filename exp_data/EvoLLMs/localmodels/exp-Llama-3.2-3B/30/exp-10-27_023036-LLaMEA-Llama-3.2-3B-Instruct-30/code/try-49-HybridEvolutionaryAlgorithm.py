import numpy as np

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitness = np.zeros(self.population_size)
        self.best_solution = self.population[np.argmin(self.fitness)]

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of the current population
            self.fitness = func(self.population)

            # Select the best individual
            self.best_solution = self.population[np.argmin(self.fitness)]

            # Perform probabilistic mutation
            mutation_prob = 0.3
            for i in range(self.population_size):
                if np.random.rand() < mutation_prob:
                    # Select a random dimension to mutate
                    dim_idx = np.random.randint(0, self.dim)
                    # Generate a new value for the dimension
                    new_val = np.random.uniform(-5.0, 5.0)
                    # Replace the old value with the new one
                    self.population[i, dim_idx] = new_val

            # Perform selection
            self.population = self.population[np.argsort(self.fitness)]

# Example usage
def func(x):
    return np.sum(x**2)

budget = 100
dim = 5
algorithm = HybridEvolutionaryAlgorithm(budget, dim)
best_solution = algorithm(func)
print("Best solution:", best_solution)