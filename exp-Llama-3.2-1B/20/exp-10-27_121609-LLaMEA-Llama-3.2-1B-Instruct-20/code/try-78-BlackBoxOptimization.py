import numpy as np

class BlackBoxOptimization:
    def __init__(self, budget, dim, mutation_rate=0.2, exploration_rate=0.2):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.exploration_rate = exploration_rate
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        # Select the best individual based on the fitness
        best_individual = self.population[np.argmax(self.fitnesses, axis=1)]

        # Perform mutation to refine the strategy
        mutated_individual = best_individual.copy()
        if np.random.rand() < self.exploration_rate:
            mutated_individual[np.random.randint(0, self.dim), np.random.randint(0, self.dim)] += np.random.uniform(-1, 1)

        # Evaluate the new individual
        new_fitness = objective(mutated_individual)

        # Update the population
        self.population[np.argmax(self.fitnesses, axis=1)] = mutated_individual
        self.fitnesses[np.argmax(self.fitnesses, axis=1), :] = new_fitness

        return mutated_individual

# Example usage:
# Create an instance of the BlackBoxOptimization class
optimizer = BlackBoxOptimization(budget=100, dim=10)

# Define a black box function
def func(x):
    return np.sin(x)

# Run the optimization algorithm
best_individual = optimizer(func)
print(best_individual)