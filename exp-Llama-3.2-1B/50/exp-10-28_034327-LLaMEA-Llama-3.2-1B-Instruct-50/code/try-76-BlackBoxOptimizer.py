import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.iterations = 0
        self.best_individual = None
        self.best_fitness = np.inf

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

    def iterated_permutation_cooling(self, func, budget):
        # Initialize the population with random points
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

        # Initialize the best individual and its fitness
        self.best_individual = population[0]
        self.best_fitness = np.max(np.max(population, axis=1))

        # Run the iterations
        for _ in range(self.iterations):
            # Evaluate the fitness of each individual
            fitness = [self.evaluate_fitness(individual, func, self.budget) for individual in population]

            # Select the fittest individuals
            self.best_individual = population[np.argmax(fitness)]

            # Create a new population by iterating over the selected individuals
            new_population = []
            for _ in range(self.budget):
                # Generate a new individual by iterating over the selected individuals
                new_individual = self.iterated_permutation_cooling(func, self.budget)
                new_population.append(new_individual)

            # Replace the old population with the new one
            population = new_population

            # Update the best individual and its fitness
            self.best_individual = population[0]
            self.best_fitness = np.max(np.max(population, axis=1))

        return self.best_individual, self.best_fitness

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
# Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# ```