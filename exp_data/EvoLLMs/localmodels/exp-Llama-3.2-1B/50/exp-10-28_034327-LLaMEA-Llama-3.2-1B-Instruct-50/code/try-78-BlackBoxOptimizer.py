import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.best_point = None
        self.best_value = -np.inf

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

    def iterated_permutation(self, func):
        # Initialize the population with random points in the search space
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

        # Evolve the population using the iterated permutation algorithm
        while self.func_evals < self.budget:
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=self.func_evals // self.budget, reverse=True)[:self.budget]

            # Create a new population by iterated permutation
            new_population = []
            for _ in range(self.budget):
                # Select a random individual from the fittest individuals
                individual = random.choice(fittest_individuals)
                # Create a new point by iterated permutation
                new_point = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
                # Evaluate the function at the new point
                new_value = func(new_point)
                # Check if the new point is within the bounds
                if -5.0 <= new_point[0] <= 5.0 and -5.0 <= new_point[1] <= 5.0:
                    # If the new point is within bounds, add it to the new population
                    new_population.append(new_point)
                    # Update the fittest individuals
                    fittest_individuals.remove(individual)
                    fittest_individuals.append(new_point)

            # Add the new population to the old population
            population.extend(new_population)

            # Update the best point and value
            self.best_point = new_population[0]
            self.best_value = np.max(func(new_population[0]))

            # Cool down the algorithm
            self.func_evals += 1

        # Return the best point found
        return self.best_point

    def iterated_cooling(self, func):
        # Initialize the population with random points in the search space
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

        # Evolve the population using the iterated cooling algorithm
        while self.func_evals < self.budget:
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=self.func_evals // self.budget, reverse=True)[:self.budget]

            # Create a new population by iterated cooling
            new_population = []
            for _ in range(self.budget):
                # Select a random individual from the fittest individuals
                individual = random.choice(fittest_individuals)
                # Create a new point by iterated cooling
                new_point = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
                # Evaluate the function at the new point
                new_value = func(new_point)
                # Check if the new point is within the bounds
                if -5.0 <= new_point[0] <= 5.0 and -5.0 <= new_point[1] <= 5.0:
                    # If the new point is within bounds, add it to the new population
                    new_population.append(new_point)
                    # Update the fittest individuals
                    fittest_individuals.remove(individual)
                    fittest_individuals.append(new_point)

            # Add the new population to the old population
            population.extend(new_population)

            # Update the best point and value
            self.best_point = new_population[0]
            self.best_value = np.max(func(new_population[0]))

            # Cool down the algorithm
            self.func_evals += 1

        # Return the best point found
        return self.best_point