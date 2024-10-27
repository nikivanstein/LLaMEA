import numpy as np

class PAEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness_values = np.zeros(self.population_size)
        self.fitness_history = []

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of the current population
            self.fitness_values = func(self.population)

            # Select the fittest individuals
            indices = np.argsort(self.fitness_values)[:self.population_size//2]
            self.population = self.population[indices]
            self.fitness_values = self.fitness_values[indices]

            # Create a new generation by applying the probability-adaptive crossover and mutation
            new_generation = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                parent1 = np.random.choice(self.population, p=self.fitness_values)
                parent2 = np.random.choice(self.population, p=self.fitness_values)
                child = parent1 + (parent2 - parent1) * np.random.uniform(0, 1, self.dim)
                new_generation[i] = child

            # Apply probability-adaptive mutation
            for i in range(self.population_size):
                if np.random.rand() < 0.3:
                    new_generation[i] += np.random.uniform(-0.5, 0.5, self.dim)

            # Replace the old population with the new generation
            self.population = new_generation

            # Store the fitness history
            self.fitness_history.append(self.fitness_values)

            # Check for convergence
            if np.all(self.fitness_values == self.fitness_values[0]):
                break

# Example usage
def func(x):
    return np.sum(x**2)

paea = PAEA(budget=100, dim=10)
paea(func)