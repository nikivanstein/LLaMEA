import numpy as np
from scipy.optimize import differential_evolution

class PADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_probability = 0.7
        self.mutation_probability = 0.1

    def __call__(self, func):
        # Initialize population with random points
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Evaluate the objective function for each point in the population
        self.values = func(self.population)

        # Repeat the process until the budget is reached
        for _ in range(self.budget):
            # Select the best point in the population
            best_idx = np.argmin(self.values)
            best_point = self.population[best_idx]

            # Select the worst point in the population
            worst_idx = np.argmax(self.values)
            worst_point = self.population[worst_idx]

            # Create a new population by applying the differential evolution operator
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                # Randomly select a point to be the parent
                parent_idx = np.random.randint(0, self.population_size)
                parent = self.population[parent_idx]

                # Randomly select a point to be the child
                child_idx = np.random.randint(0, self.population_size)
                child = self.population[child_idx]

                # Apply the crossover operator
                if np.random.rand() < self.crossover_probability:
                    child = self.crossover(parent, child)

                # Apply the mutation operator
                if np.random.rand() < self.mutation_probability:
                    child = self.mutation(child)

                # Replace the worst point in the population with the child
                if i == worst_idx:
                    new_population[i] = child
                    self.values[i] = func(child)
                else:
                    new_population[i] = child
                    self.values[i] = func(child)

            # Update the population
            self.population = new_population

            # Evaluate the objective function for each point in the population
            self.values = func(self.population)

        # Return the best point in the final population
        return np.argmin(self.values)

    def crossover(self, parent, child):
        # Perform a simple crossover by taking the average of the two parents
        return (parent + child) / 2

    def mutation(self, point):
        # Perform a simple mutation by adding a random value to the point
        return point + np.random.uniform(-1.0, 1.0)